"""Role-based access control helpers."""

from __future__ import annotations

from material_intelligence.app.db.models import User
from material_intelligence.app.domain.enums import UserRole

ROLE_PERMISSIONS: dict[UserRole, set[str]] = {
    UserRole.ADMINISTRATOR: {
        "view",
        "inspect",
        "manage_datasets",
        "train",
        "evaluate",
        "approve",
        "manage_inventory",
        "admin",
        "reports",
    },
    UserRole.AI_RESEARCHER: {
        "view",
        "inspect",
        "manage_datasets",
        "train",
        "evaluate",
        "reports",
    },
    UserRole.QUALITY_ENGINEER: {
        "view",
        "inspect",
        "approve",
        "manage_inventory",
        "reports",
    },
    UserRole.SITE_SUPERVISOR: {
        "view",
        "inspect",
        "manage_inventory",
        "reports",
    },
    UserRole.VIEWER: {"view", "reports"},
}


class AuthorizationError(PermissionError):
    """Raised when a user lacks the required permission."""


def user_permissions(user: User | None) -> set[str]:
    if user is None:
        return set()
    try:
        role = UserRole(user.role)
    except ValueError:
        return set()
    return ROLE_PERMISSIONS.get(role, set())


def has_permission(user: User | None, permission: str) -> bool:
    return permission in user_permissions(user)


def require_permission(user: User | None, permission: str) -> None:
    if not has_permission(user, permission):
        raise AuthorizationError(f"Permission '{permission}' is required.")
