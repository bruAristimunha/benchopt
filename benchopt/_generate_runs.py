import copy
import inspect
import itertools

from .utils.parametrized_name_mixin import is_matched
from .utils.run_context import RunContext

# Canonical loop nesting order when no grouping is requested (rep innermost).
CANONICAL_AXES = ('dataset', 'objective', 'solver')
# `repetition` is groupable but needs the generator hoist to nest above solver.
GROUPABLE_AXES = CANONICAL_AXES + ('repetition',)


def _normalize_group_by(group_by):
    """Validate ``group_by`` and return it as a list.

    ``repetition`` is a valid axis, but its cardinality is resolved per
    (dataset, objective) (possibly from ``objective.cv``, where each rep is a
    different fold), so it must nest *inside* ``dataset`` and ``objective`` --
    i.e. both must also be in ``group_by``. It is otherwise free relative to
    ``solver``: ``[dataset, objective, repetition]`` batches solvers under a
    fold (sharing per-fold setup); ``[dataset, objective, solver]`` batches
    reps under a config (sharing solver state).
    """
    if not group_by:
        return []
    if isinstance(group_by, str):
        group_by = [group_by]
    group_by = list(group_by)
    unknown = [a for a in group_by if a not in GROUPABLE_AXES]
    if unknown:
        raise ValueError(
            f"Unknown `group_by` axes {unknown}. Valid axes: {GROUPABLE_AXES}."
        )
    if len(set(group_by)) != len(group_by):
        raise ValueError(f"Duplicate axes in `group_by`: {group_by}.")
    if 'repetition' in group_by:
        # 'dataset' and 'objective' must not only be present but nest *outside*
        # rep (the fold depends on them), i.e. appear before it in the order.
        outer = set(group_by[:group_by.index('repetition')])
        if not {'dataset', 'objective'} <= outer:
            raise ValueError(
                "'repetition' in `group_by` must nest inside 'dataset' and "
                "'objective' (both must appear before it), since the fold "
                f"depends on them. Got {group_by}."
            )
    return group_by


def _ordered_axes(group_by):
    """Loop nesting order with the ``group_by`` axes outermost.

    ``group_by=None`` keeps the canonical ``dataset -> objective -> solver``
    order, so the default run stream is unchanged. A str or list of axes moves
    those axes to the front (outermost), making runs contiguous by that key.

    The order also drives memory: only the outermost axis is streamed (see
    :func:`_materialize_axes`), so list the heaviest, data-carrying axis first.
    """
    group_by = _normalize_group_by(group_by)
    # Remaining axes keep canonical order; rep stays innermost by default.
    rest = [a for a in GROUPABLE_AXES if a not in group_by]
    return group_by + rest


# `prepare` kwargs carry no `meta`, hence `.get('meta', {})` + fallback.
_AXIS_KEY = {
    'dataset': lambda kw: kw.get('meta', {}).get('dataset_name',
                                                 kw.get('dataset')),
    'objective': lambda kw: kw.get('meta', {}).get('objective_name',
                                                   kw.get('objective')),
    'solver': lambda kw: kw.get('meta', {}).get('solver_name',
                                                kw.get('solver')),
    'repetition': lambda kw: kw.get('meta', {}).get('idx_rep'),
}


def group_runs(run_kwargs_iter, group_by):
    """Yield contiguous batches of run kwargs sharing the ``group_by`` key.

    Relies on the generator emitting runs already ordered by ``group_by`` (see
    :func:`_ordered_axes`), so grouping is a lazy ``itertools.groupby`` with no
    full materialization. ``group_by=None`` yields one run per batch, i.e. the
    default one job per (dataset, objective, solver, repetition). Any config
    grouping collapses that config's repetitions into a single batch (the #860
    goal), since repetitions are always the innermost axis.
    """
    group_by = _normalize_group_by(group_by)
    if not group_by:
        for run_kwargs in run_kwargs_iter:
            yield [run_kwargs]
        return
    keys = [_AXIS_KEY[axis] for axis in group_by]

    def _key(run_kwargs):
        return tuple(k(run_kwargs) for k in keys)

    for _, batch in itertools.groupby(run_kwargs_iter, key=_key):
        yield list(batch)


def _enter_axis(terminal, axis, item, is_installed, i_solver=None):
    """Run terminal side-effects for entering `item`; return False to skip.

    ``i_solver`` is the solver's position in its axis loop; its only use is
    gating ``terminal.skip`` so an objective-skip prints once per (dataset,
    objective) -- so it is passed only for the solver axis.
    """
    if axis == 'dataset':
        terminal.set(dataset=item)
        if not is_installed:
            terminal.show_status('not installed', dataset=True)
            return False
        terminal.display_dataset()
    elif axis == 'objective':
        terminal.set(objective=item)
        if not is_installed:
            terminal.show_status('not installed', objective=True)
            return False
        terminal.display_objective()
    else:  # solver
        terminal.set(solver=item, i_solver=i_solver)
        if not is_installed:
            terminal.show_status('not installed')
            return False
    return True


def _materialize_axes(order, datasets, objectives, solvers):
    """Build each axis's instances, keeping only the outermost lazy.

    The outermost axis (``order[0]``) is iterated once, so it is left as a lazy
    generator -- as in the ungrouped default on ``main``: its instances, and
    the data they cache (e.g. ``Dataset._data``), are freed as the nest
    advances instead of all being held resident. The inner axes are re-entered
    per outer combination, so they are materialized (listed) and stay resident
    for the whole run.

    Ordering therefore matters for memory: only the *outermost* axis is
    streamed. Put the heaviest, data-carrying axis (usually ``dataset``) first
    in ``group_by`` so its data is freed as the run advances -- a heavy axis
    placed after the first group_by axis keeps every one of its instances (and
    their cached data) resident. E.g. ``group_by=['dataset', 'objective']``
    streams datasets, whereas ``['objective', 'dataset']`` holds them all.
    """
    from .benchmark import _list_parametrized_classes
    gens = {
        'dataset': _list_parametrized_classes(*datasets),
        'objective': _list_parametrized_classes(*objectives,
                                                check_installed=False),
        'solver': _list_parametrized_classes(*solvers),
    }
    return {axis: (g if axis == order[0] else list(g))
            for axis, g in gens.items()}


def _run_meta(benchmark, dataset, objective, solver, rep, sampling_strategy,
              obj_description):
    """Build the per-run ``meta`` dict (feeds display and the cache key)."""
    return {
        'base_seed': benchmark.seed,
        'objective_name': str(objective),
        'obj_description': obj_description,
        'solver_name': str(solver),
        'solver_description': inspect.cleandoc(solver.__doc__ or ""),
        'dataset_name': str(dataset),
        'idx_rep': rep,
        'sampling_strategy': sampling_strategy.capitalize(),
        'file_objective': objective._module_filename.name,
        **{f"p_obj_{k}": v for k, v in objective._parameters.items()},
        'file_solver': f"solvers/{solver._module_filename.name}",
        **{f"p_solver_{k}": v for k, v in solver._parameters.items()},
        'file_dataset': f"datasets/{dataset._module_filename.name}",
        **{f"p_dataset_{k}": v for k, v in dataset._parameters.items()},
    }


def _get_all_runs(benchmark, solvers=None, forced_solvers=None,
                  datasets=None, objectives=None, terminal=None,
                  group_by=None):
    """Generator with all combinations to run for the benchmark.

    Parameters
    ----------
    benchmark : benchopt.Benchmark object
        Object to represent the benchmark.
    solvers : list | None
        List of solvers to include in the benchmark. If None
        all solvers available are run.
    forced_solvers : list | None
        List of solvers to include in the benchmark and for
        which one forces recomputation.
    datasets : list | None
        List of datasets to include. If None all available
        datasets are used.
    objectives : list | None
        Filters to select specific objective parameters. If None,
        all objective parameters are tested
    terminal : TerminalOutput or None
        Object to format string to display the terminal.

    Yields
    ------
    dataset : BaseDataset instance
    objective : BaseObjective instance
    solver : BaseSolver instance
    force : bool
    """
    # Non-rep path: rep is expanded downstream by `get_solver_kwargs`, so only
    # nest the d/o/s axes here (rep is always last in the order when absent).
    order = [a for a in _ordered_axes(group_by) if a != 'repetition']
    axis_items = _materialize_axes(order, datasets, objectives, solvers)
    terminal.set_levels(order)

    def _rec(depth, chosen):
        axis = order[depth]
        for i_axis, (item, is_installed) in enumerate(axis_items[axis]):
            if not _enter_axis(terminal, axis, item, is_installed,
                               i_solver=i_axis):
                continue
            chosen[axis] = item
            if depth + 1 == len(order):
                solver = chosen['solver']
                yield dict(
                    dataset=chosen['dataset'], objective=chosen['objective'],
                    solver=solver,
                    force=is_matched(
                        str(solver), forced_solvers, default=False
                    ),
                    terminal=terminal,
                )
            else:
                yield from _rec(depth + 1, chosen)

    yield from _rec(0, {})


def get_solver_kwargs(
    benchmark, dataset, objective, solver, n_repetitions, max_runs,
    timeout=None, force=False, collect=False, terminal=None,
    run_context=None,
):
    """Run a benchmark for a given dataset, objective and solver.

    Parameters
    ----------
    benchmark : benchopt.Benchmark object
        Object to represent the benchmark.
    dataset : instance of BaseDataset
        The dataset used for this benchmark.
    objective : instance of BaseObjective
        The objective to minimize.
    solver : instance of BaseSolver
        The solver to use.
    n_repetitions : int
        The number of repetitions to run. Defaults to 1.
    max_runs : int
        The maximum number of solver runs to perform to estimate
        the convergence curve.
    timeout : float
        The maximum duration in seconds of the solver run.
    force : bool
        If force is set to True, ignore the cache and run the computations
        for the solver anyway. Else, use the cache if available.
    collect : bool
        If set to True, only collect the results that have been put in cache,
        and ignore the results that are not computed yet, default is False.
    terminal : TerminalOutput or None
        Object to format string to display the progress of the solver.
    run_context : RunContext | None
        Base context created in ``_run_benchmark`` carrying config fields
        (``pdb``, ``run_output_base``).  Cloned here for each repetition
        with the per-run fields filled in.

    Returns
    -------
    args_run_one_to_cvg : dict
        The dictionary of arguments to run_one_to_cvg.
    """
    run_context = run_context or RunContext()

    # Resolve inheritance now rather than at `_set_objective` run time: meta
    # below feeds the cache key, so it must not depend on whether some other
    # (solver, dataset) pair already triggered this resolution earlier on.
    solver._inherit_stopping_criterion(objective)

    # get sampling strategy
    # for plotting purpose consider 'callback' as 'iteration'
    sampling_strategy = solver._solver_strategy
    if sampling_strategy == 'callback':
        sampling_strategy = 'iteration'

    # get objective description
    # use `obj_` instead of `objective_` to avoid conflicts with
    # the name of metrics in Objective.compute
    obj_description = objective.__doc__ or ""

    # Set run context with rep=0 so get_seed() works during _set_dataset.
    run_context = run_context.set_run_context(
        objective=objective, dataset=dataset, solver=solver,
        repetition=0, base_seed=benchmark.seed,
    )

    # Set objective and skip if necessary. This prepares the base objective at
    # fold 0; mark it so rep-0 copies reuse it instead of re-preparing.
    skip, reason = objective._set_dataset(dataset)
    if skip:
        terminal.skip(reason, objective=True)
        return []
    objective._prepared_rep = 0

    if n_repetitions is None:
        if hasattr(objective, "cv"):
            n_repetitions = objective.cv.get_n_splits(
                **getattr(objective, "cv_metadata", {})
            )
        else:
            # we set 1 by default so that the solver run at least once
            n_repetitions = 1

    terminal.n_repetitions = n_repetitions

    for rep in range(n_repetitions):
        objective_rep = copy.copy(objective)

        # Get meta
        meta = _run_meta(benchmark, dataset, objective, solver, rep,
                         sampling_strategy, obj_description)

        # The repetition travels in the run context; the compute node pulls it
        # and prepares the fold (see `BaseObjective._set_repetition`), so we do
        # not stamp the rep or build every fold on the front node.
        run_ctx = run_context.set_run_context(
            objective=objective_rep, dataset=dataset, solver=solver,
            repetition=rep, base_seed=benchmark.seed,
        )

        args_run_one_to_cvg = dict(
            benchmark=benchmark, objective=objective_rep, solver=solver,
            meta=meta, timeout=timeout, max_runs=max_runs, force=force,
            terminal=terminal, run_context=run_ctx,
        )

        yield args_run_one_to_cvg


def _prepare_folds(benchmark, base_run_context, dataset, objective,
                   n_repetitions, terminal):
    """Enumerate the per-fold objectives shared across solvers of a (d, o).

    Returns a list of ``(rep, objective_rep)`` -- one shared objective per
    fold, *not yet prepared* -- or ``None`` if the (dataset, objective) pair
    is skipped. Each fold's data is prepared lazily on the compute node by the
    first solver (see :meth:`BaseObjective._set_repetition`) and reused by the
    others, so no fold is built up front on the front node.
    """
    # rep=0 context so get_seed works during the skip-check _set_dataset.
    base_run_context.set_run_context(
        objective=objective, dataset=dataset, solver=None, repetition=0,
        base_seed=benchmark.seed,
    )
    skip, reason = objective._set_dataset(dataset)
    if skip:
        terminal.skip(reason, objective=True)
        return None

    if n_repetitions is None:
        if hasattr(objective, "cv"):
            n_repetitions = objective.cv.get_n_splits(
                **getattr(objective, "cv_metadata", {})
            )
        else:
            n_repetitions = 1
    terminal.n_repetitions = n_repetitions

    folds = []
    for rep in range(n_repetitions):
        objective_rep = copy.copy(objective)
        objective_rep._repetition = rep
        folds.append((rep, objective_rep))
    return folds


def _generate_rep_grouped(benchmark, order, axis_items, forced_solvers,
                          n_repetitions, max_runs, timeout, terminal,
                          run_context):
    """Ordered nest over d/o/rep/s, sharing a fold's objective across solvers.

    Used when ``repetition`` is a group_by axis (e.g. ``[dataset, objective,
    repetition]`` batches solvers under each fold). Yields final run kwargs.
    The nest over ``order`` is walked as a recursive program (``_rec``), not
    fixed loops, so any axis ordering works.
    """
    base_ctx = run_context or RunContext()
    terminal.set_levels(order)
    # `dataset` and `objective` are always outer to `repetition` (enforced by
    # `_normalize_group_by`), so a (d, o) is fully emitted before the next one.
    # Cache only the current (d, o)'s folds: advancing to the next (d, o) drops
    # the previous folds so their prepared data can be GC'd once dispatched. A
    # persistent cache would instead pin every fold of every (d, o) all run.
    current_folds = {'key': None, 'folds': None}

    def folds_for(dataset, objective):
        key = (id(dataset), id(objective))
        if current_folds['key'] != key:
            current_folds['key'] = key
            current_folds['folds'] = _prepare_folds(
                benchmark, base_ctx, dataset, objective, n_repetitions,
                terminal,
            )
        return current_folds['folds']

    def _leaf(chosen):
        rep, objective_rep = chosen['repetition']
        dataset, objective, solver = (
            chosen['dataset'], chosen['objective'], chosen['solver']
        )
        solver._inherit_stopping_criterion(objective)
        sampling_strategy = solver._solver_strategy
        if sampling_strategy == 'callback':
            sampling_strategy = 'iteration'
        meta = _run_meta(benchmark, dataset, objective, solver, rep,
                         sampling_strategy, objective.__doc__ or "")
        run_ctx = base_ctx.set_run_context(
            objective=objective_rep, dataset=dataset, solver=solver,
            repetition=rep, base_seed=benchmark.seed,
        )
        return dict(
            benchmark=benchmark, objective=objective_rep, solver=solver,
            meta=meta, timeout=timeout, max_runs=max_runs,
            force=is_matched(str(solver), forced_solvers, default=False),
            terminal=terminal, run_context=run_ctx,
        )

    def _rec(depth, chosen):
        axis = order[depth]
        last = depth + 1 == len(order)
        if axis == 'repetition':
            folds = folds_for(chosen['dataset'], chosen['objective'])
            if folds is None:  # skipped (dataset, objective)
                return
            items = [(f, True) for f in folds]
        else:
            items = axis_items[axis]
        for i_axis, (item, is_installed) in enumerate(items):
            if axis == 'repetition':
                terminal.display_rep(item[0])  # item = (rep, objective_rep)
            elif not _enter_axis(terminal, axis, item, is_installed,
                                 i_solver=i_axis):
                continue
            chosen[axis] = item
            if last:
                yield _leaf(chosen)
            else:
                yield from _rec(depth + 1, chosen)

    yield from _rec(0, {})


def generate_run_kwargs(
    benchmark, solvers=None, forced_solvers=None, datasets=None,
    objectives=None, n_repetitions=1, max_runs=10, timeout=None,
    collect=False, terminal=None, run_context=None, group_by=None,
):
    """Yield kwargs for each ``run_one_to_cvg`` call in the benchmark.

    Combines the (dataset, objective, solver) enumeration with the per-run
    metadata so that callers only need a single generator to drive the
    benchmark execution.
    """
    if 'repetition' in _normalize_group_by(group_by):
        # rep is a grouping axis -> nest it (above solver) and share each fold
        # across solvers. A separate path (not merged with the default) because
        # fold-sharing runs `_set_dataset` once per (dataset, objective) with
        # `solver=None`, whereas the default runs it per (d, o, solver); a
        # merge would shift the default seed context and break byte-identity.
        order = _ordered_axes(group_by)
        axis_items = _materialize_axes(order, datasets, objectives, solvers)
        yield from _generate_rep_grouped(
            benchmark, order, axis_items,
            forced_solvers, n_repetitions, max_runs, timeout, terminal,
            run_context,
        )
        return

    all_runs = _get_all_runs(
        benchmark, solvers, forced_solvers, datasets, objectives,
        terminal=terminal, group_by=group_by,
    )
    common_kwargs = dict(
        benchmark=benchmark, n_repetitions=n_repetitions, max_runs=max_runs,
        timeout=timeout, collect=collect, run_context=run_context,
    )
    for kwargs in all_runs:
        yield from get_solver_kwargs(**common_kwargs, **kwargs)
