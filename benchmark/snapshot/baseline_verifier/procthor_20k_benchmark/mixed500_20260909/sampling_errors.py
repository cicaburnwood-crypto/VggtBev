"""Only pre-episode scene/task infeasibility is eligible for replacement."""
class SceneSamplingError(RuntimeError):
    pass

LEGACY_EMPTY_GRAPH = "RuntimeError('reachable graph has no useful connected component')"

def legacy_sampling_failure(error, has_task, has_results):
    return (error == LEGACY_EMPTY_GRAPH and not has_task and not has_results)
