"""
Dashboard pages package.

This package contains all the pages for the ISRO GNSS Dashboard.
"""
from .dashboard import dashboard_page, DashboardPage
from .satellites import satellites_page, SatellitesPage
from .alerts import alerts_page, AlertsPage
from .admin import admin_panel, AdminPanel

# Explicitly specify what gets imported with 'from package import *'
__all__ = [
    'dashboard_page', 'DashboardPage',
    'satellites_page', 'SatellitesPage',
    'alerts_page', 'AlertsPage',
    'admin_panel', 'AdminPanel'
]
