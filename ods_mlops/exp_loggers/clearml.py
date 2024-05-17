from typing import Any, Dict, Optional, Union

from clearml import Task

from .base_logger import BaseExpLogger


class ClearMLExpLogger(BaseExpLogger):
    def __init__(self, exp_name: str, run_name: Optional[str] = None):
        super().__init__(exp_name, run_name)
        self.task = None

    def init(self):
        self.task: Task = Task.init(
            project_name=self.exp_name,
            task_name=self.run_name,
            auto_connect_arg_parser=False,
            auto_connect_frameworks=False,
            auto_resource_monitoring=False,
            auto_connect_streams=False,
            reuse_last_task_id=False,
        )

    def log_params(self, params: Dict[str, Any]):
        self.task.connect(params)

    def log_metric(self, key: str, value: Union[int, float]):
        self.task.get_logger().report_single_value(key, value=value)

    def log_figure(self, figure, name):
        self.task.get_logger().report_matplotlib_figure(
            title=name, series=name, figure=figure, iteration=None
        )

    def close(self):
        self.task.close()
