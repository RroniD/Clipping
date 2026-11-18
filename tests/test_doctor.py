"""Setup checker: plain-English report, never prints keys."""

from clipper import doctor


def test_report_marks_problems_and_fixes():
    checks = [doctor.Check(True, "ffmpeg", "found"),
              doctor.Check(False, "AI key", "missing", "Paste it into .env"),
              doctor.Check(False, "Phone notifications", "not set up", "Run the setup", important=False)]
    text = doctor.report(checks)
    assert "OK  ffmpeg: found" in text
    assert "!!  AI key: missing" in text and "-> Paste it into .env" in text
    assert "--  Phone notifications" in text
    assert "1 thing(s) need fixing" in text


def test_offline_checks_run_without_printing_keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-value")
    text = doctor.report(doctor.run_checks(online=False))
    assert "sk-ant-secret-value" not in text and "AI key" in text