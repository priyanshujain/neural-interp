from abc import ABC, abstractmethod
from typing import Any

from markupsafe import Markup

from neural_interp.report._context import ReportContext
from neural_interp.report._render import render
from neural_interp.report._views import GuideEntry


class Section(ABC):
    title: str = ""
    template: str = ""
    guide: tuple[GuideEntry, ...] = ()

    def visible(self, context: ReportContext) -> bool:
        return True

    def hits(self, context: ReportContext) -> set[int]:
        return set()

    def write_assets(self, context: ReportContext) -> None:
        return None

    @abstractmethod
    def view(self, context: ReportContext) -> dict[str, Any]: ...

    def render(self, context: ReportContext, step: int) -> Markup:
        return Markup(render(self.template, step=step, title=self.title, **self.view(context)))
