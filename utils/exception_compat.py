"""Keep secondary failure context on Python 3.10 and newer."""


def add_exception_note(error, message):
    add_note = getattr(error, "add_note", None)
    if add_note is not None:
        add_note(message)
        return
    notes = getattr(error, "__notes__", None)
    if notes is None:
        notes = []
        error.__notes__ = notes
    notes.append(message)
