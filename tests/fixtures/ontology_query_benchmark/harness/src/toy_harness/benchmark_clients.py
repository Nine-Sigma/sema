from toy_harness.benchmark_models import Execution
from toy_harness.benchmark_workflow import sanitize_message


def failed_execution(exc: BaseException, latency_s: float) -> Execution:
    return Execution(False, (), (), latency_s, "sql_error", sanitize_message(str(exc)))
