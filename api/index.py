"""Vercel serverless entry point — exposes the existing Flask application."""

from webapp.app import app  # noqa: E402

# Vercel Python runtime picks up the WSGI callable named `app`.
app = app  # noqa: PLW0127