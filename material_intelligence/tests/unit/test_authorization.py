"""Authorization rule tests."""

from __future__ import annotations

import pytest

from material_intelligence.app.db.models import User
from material_intelligence.app.domain.enums import UserRole
from material_intelligence.app.security.authorization import (
    AuthorizationError,
    has_permission,
    require_permission,
)


def _user(role: UserRole) -> User:
    return User(
        email=f"{role.value}@example.com",
        full_name="Test",
        password_hash="x",
        role=role.value,
        is_active=True,
    )


def test_viewer_cannot_train():
    user = _user(UserRole.VIEWER)
    assert has_permission(user, "view")
    assert not has_permission(user, "train")
    with pytest.raises(AuthorizationError):
        require_permission(user, "approve")


def test_admin_has_all_core_permissions():
    user = _user(UserRole.ADMINISTRATOR)
    for perm in ("view", "inspect", "train", "approve", "admin"):
        assert has_permission(user, perm)
