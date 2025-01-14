from shiny.playwright import controller
from shiny.run import ShinyAppProc
from playwright.sync_api import Page
from shiny.pytest import create_app_fixture

app = create_app_fixture("../dashboard/app.py")

def test_basic_app(page: Page, app: ShinyAppProc):
    page.goto(app.url)

    assert "Foreshadownba prediction dashboard" in page.title(), "Dashboard title mismatch"

    # txt = controller.OutputText(page, "txt")
    # slider = controller.InputSlider(page, "n")
    # slider.set("55")
    # txt.expect_value("n*2 is 110")
