"""Visual theme — BuildShield industrial (black + high-vis orange/yellow)."""

from __future__ import annotations

from nicegui import ui

# BuildShield direction: bold, industrial, confident — black with orange-yellow.
LIGHT = {
    "bg": "#eceae6",
    "surface": "#f5f3ef",
    "surface_alt": "#e4e1db",
    "border": "#cfc8bc",
    "text": "#111111",
    "muted": "#5c574f",
    "accent": "#e8940a",
    "accent_dim": "#c47808",
    "accent_soft": "rgba(232,148,10,0.12)",
    "success": "#1f8f63",
    "warning": "#d97706",
    "danger": "#c53d3d",
    "card": "#ffffff",
    "topbar": "rgba(245,243,239,0.94)",
    "sidebar": "#0d0d0d",
    "sidebar_text": "#f2f0eb",
    "sidebar_muted": "#9a948a",
    "body_bg": (
        "radial-gradient(800px 380px at 100% 0%, rgba(232,148,10,0.14) 0%, transparent 50%),"
        "linear-gradient(165deg, #e4e1db 0%, #eceae6 45%, #f5f3ef 100%)"
    ),
    "pattern": (
        "linear-gradient(rgba(17,17,17,0.035) 1px, transparent 1px),"
        "linear-gradient(90deg, rgba(17,17,17,0.035) 1px, transparent 1px)"
    ),
}

DARK = {
    "bg": "#0a0a0a",
    "surface": "#121212",
    "surface_alt": "#1a1a1a",
    "border": "#2e2e2e",
    "text": "#f5f3ef",
    "muted": "#a39e94",
    "accent": "#f5a623",
    "accent_dim": "#e8940a",
    "accent_soft": "rgba(245,166,35,0.16)",
    "success": "#3ecf8e",
    "warning": "#f5a623",
    "danger": "#ef5d5d",
    "card": "#141414",
    "topbar": "rgba(18,18,18,0.94)",
    "sidebar": "#000000",
    "sidebar_text": "#f5f3ef",
    "sidebar_muted": "#8a857c",
    "body_bg": (
        "radial-gradient(900px 480px at 90% -10%, rgba(245,166,35,0.18) 0%, transparent 48%),"
        "linear-gradient(180deg, #121212 0%, #0a0a0a 60%)"
    ),
    "pattern": (
        "linear-gradient(rgba(245,243,239,0.03) 1px, transparent 1px),"
        "linear-gradient(90deg, rgba(245,243,239,0.03) 1px, transparent 1px)"
    ),
}


def apply_theme(mode: str = "light") -> None:
    colors = DARK if mode == "dark" else LIGHT
    if mode == "dark":
        ui.dark_mode().enable()
    else:
        ui.dark_mode().disable()

    ui.colors(
        primary=colors["accent"],
        secondary=colors["accent_dim"],
        accent=colors["accent"],
        dark=colors["bg"],
        positive=colors["success"],
        negative=colors["danger"],
        info=colors["accent"],
        warning=colors["warning"],
    )
    ui.add_head_html(
        f"""
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=Barlow:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

          :root {{
            --mi-bg: {colors["bg"]};
            --mi-surface: {colors["surface"]};
            --mi-surface-alt: {colors["surface_alt"]};
            --mi-border: {colors["border"]};
            --mi-text: {colors["text"]};
            --mi-muted: {colors["muted"]};
            --mi-accent: {colors["accent"]};
            --mi-accent-dim: {colors["accent_dim"]};
            --mi-accent-soft: {colors["accent_soft"]};
            --mi-card: {colors["card"]};
            --mi-success: {colors["success"]};
            --mi-warning: {colors["warning"]};
            --mi-danger: {colors["danger"]};
            --mi-sidebar: {colors["sidebar"]};
            --mi-sidebar-text: {colors["sidebar_text"]};
            --mi-sidebar-muted: {colors["sidebar_muted"]};
            --mi-radius: 4px;
            --mi-shadow: 0 1px 0 rgba(0,0,0,0.04), 0 10px 28px rgba(0,0,0,0.08);
          }}

          * {{ box-sizing: border-box; }}
          body {{
            font-family: 'Barlow', sans-serif;
            background: {colors["body_bg"]};
            color: var(--mi-text);
            letter-spacing: 0.01em;
          }}
          body::before {{
            content: '';
            position: fixed;
            inset: 0;
            pointer-events: none;
            background-image: {colors["pattern"]};
            background-size: 24px 24px;
            opacity: 0.7;
            z-index: 0;
          }}
          .mi-shell {{ min-height: 100vh; position: relative; z-index: 1; }}
          .mi-shell.mi-lock-scroll {{ height: 100vh; max-height: 100vh; overflow: hidden; }}

          .mi-sidebar {{
            width: 248px;
            transition: width 180ms ease;
            background: var(--mi-sidebar);
            border-right: 1px solid rgba(245,166,35,0.22);
            color: var(--mi-sidebar-text);
          }}
          .mi-sidebar.collapsed {{ width: 76px; }}
          .mi-sidebar.collapsed .mi-nav-label,
          .mi-sidebar.collapsed .mi-brand-sub,
          .mi-sidebar.collapsed .mi-brand-line {{ display: none; }}

          .mi-brand {{
            font-family: 'Oswald', sans-serif;
            font-weight: 700;
            letter-spacing: 0.12em;
            font-size: 0.82rem;
            color: #f5f3ef;
            text-transform: uppercase;
          }}
          .mi-brand-mark {{
            width: 36px; height: 36px;
            background: var(--mi-accent);
            color: #0d0d0d;
            display: grid; place-items: center;
            font-family: 'Oswald', sans-serif;
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            flex-shrink: 0;
          }}
          .mi-nav-item {{
            border-radius: 2px;
            color: var(--mi-sidebar-muted);
            transition: background 140ms ease, color 140ms ease, border-color 140ms ease;
            display: flex;
            align-items: center;
            gap: 0.65rem;
            text-decoration: none !important;
            border-left: 3px solid transparent;
            padding-left: 0.7rem !important;
          }}
          .mi-nav-item:hover {{
            background: rgba(245,166,35,0.10);
            color: #f5f3ef;
          }}
          .mi-nav-item.active {{
            background: rgba(245,166,35,0.16);
            color: var(--mi-accent);
            border-left-color: var(--mi-accent);
            font-weight: 600;
          }}

          .mi-topbar {{
            background: {colors["topbar"]};
            backdrop-filter: blur(10px);
            border-bottom: 1px solid var(--mi-border);
          }}
          .mi-page-title {{
            font-family: 'Oswald', sans-serif;
            font-weight: 600;
            font-size: 1.35rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
          }}

          .mi-panel {{
            background: var(--mi-card);
            border: 1px solid var(--mi-border);
            border-radius: var(--mi-radius);
            box-shadow: var(--mi-shadow);
          }}
          .mi-card {{
            background: var(--mi-card);
            border: 1px solid var(--mi-border);
            border-radius: var(--mi-radius);
            box-shadow: var(--mi-shadow);
            transition: border-color 160ms ease, transform 160ms ease;
          }}
          .mi-card:hover {{ border-color: color-mix(in srgb, var(--mi-accent) 45%, var(--mi-border)); }}

          .mi-kpi {{
            position: relative;
            overflow: hidden;
            padding: 1.05rem 1.1rem 1rem;
            animation: miFadeUp 420ms ease both;
          }}
          .mi-kpi::before {{
            content: '';
            position: absolute;
            left: 0; top: 0; bottom: 0;
            width: 4px;
            background: var(--mi-accent);
          }}
          .mi-kpi-label {{
            font-size: 0.72rem;
            font-weight: 700;
            color: var(--mi-muted);
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-family: 'Oswald', sans-serif;
          }}
          .mi-metric-value {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 1.75rem;
            font-weight: 600;
            color: var(--mi-text);
            line-height: 1.15;
            margin-top: 0.35rem;
          }}
          .mi-delta-up {{ color: var(--mi-success); font-size: 0.75rem; font-weight: 600; }}
          .mi-delta-down {{ color: var(--mi-danger); font-size: 0.75rem; font-weight: 600; }}
          .mi-muted {{ color: var(--mi-muted); }}

          .mi-hero {{
            position: relative;
            overflow: hidden;
            padding: 1.6rem 1.7rem;
            background:
              linear-gradient(115deg, rgba(245,166,35,0.18) 0%, transparent 38%),
              linear-gradient(180deg, #1a1a1a 0%, #0d0d0d 100%);
            border: 1px solid rgba(245,166,35,0.35);
            border-radius: 4px;
            color: #f5f3ef;
            animation: miFadeUp 480ms ease both;
          }}
          .mi-hero .mi-muted {{ color: #b8b2a6 !important; }}
          .mi-hero-brand {{
            font-family: 'Oswald', sans-serif;
            font-size: clamp(1.85rem, 3vw, 2.55rem);
            font-weight: 700;
            letter-spacing: 0.02em;
            line-height: 1.05;
            text-transform: uppercase;
            color: #f5f3ef;
          }}
          .mi-hero-brand span {{ color: var(--mi-accent); }}
          .mi-hero-rule {{
            width: 64px; height: 4px;
            background: var(--mi-accent);
            margin: 0.75rem 0 0.85rem;
          }}
          .mi-hero-stripe {{
            position: absolute;
            right: -40px; top: -20px;
            width: 180px; height: 220%;
            background: repeating-linear-gradient(
              -18deg,
              transparent,
              transparent 10px,
              rgba(245,166,35,0.08) 10px,
              rgba(245,166,35,0.08) 20px
            );
            pointer-events: none;
          }}

          .mi-section {{
            display: flex;
            flex-direction: column;
            gap: 0.85rem;
            animation: miFadeUp 520ms ease both;
          }}
          .mi-section-title {{
            font-family: 'Oswald', sans-serif;
            font-size: 1.05rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            display: flex;
            align-items: center;
            gap: 0.55rem;
          }}
          .mi-section-title::before {{
            content: '';
            width: 12px; height: 12px;
            background: var(--mi-accent);
            flex-shrink: 0;
          }}
          .mi-section-sub {{
            font-size: 0.86rem;
            color: var(--mi-muted);
            margin-top: -0.35rem;
            margin-left: 1.35rem;
          }}

          .mi-filter-bar {{
            background: var(--mi-card);
            border: 1px solid var(--mi-border);
            border-left: 4px solid var(--mi-accent);
            border-radius: var(--mi-radius);
            padding: 0.85rem 1rem;
            box-shadow: var(--mi-shadow);
          }}

          .mi-status-dot {{
            width: 8px; height: 8px; border-radius: 50%;
            display: inline-block; margin-right: 6px;
          }}
          .mi-status-ok {{ background: var(--mi-success); box-shadow: 0 0 0 3px rgba(31,143,99,0.15); }}
          .mi-status-warn {{ background: var(--mi-warning); }}
          .mi-status-err {{ background: var(--mi-danger); }}

          .mi-empty {{
            border: 1px dashed var(--mi-border);
            border-radius: var(--mi-radius);
            padding: 1.6rem;
            text-align: center;
            color: var(--mi-muted);
            background: var(--mi-surface-alt);
          }}
          .mi-chip {{
            display: inline-flex;
            align-items: center;
            border-radius: 2px;
            padding: 0.12rem 0.45rem;
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            border: 1px solid var(--mi-border);
            color: var(--mi-muted);
            background: var(--mi-surface-alt);
            font-family: 'Oswald', sans-serif;
          }}
          .mi-chip-demo {{
            border-color: color-mix(in srgb, var(--mi-accent) 55%, var(--mi-border));
            color: var(--mi-accent-dim);
            background: var(--mi-accent-soft);
          }}
          .mi-thumb {{
            width: 52px; height: 52px; object-fit: cover;
            border-radius: 2px; border: 1px solid var(--mi-border);
          }}
          .mi-kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
            gap: 0.85rem;
            width: 100%;
          }}
          .mi-chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
            gap: 0.9rem;
            width: 100%;
          }}
          .mi-insp-row {{
            transition: background 140ms ease;
            border-radius: 2px;
            padding: 0.55rem 0.4rem;
          }}
          .mi-insp-row:hover {{ background: var(--mi-accent-soft); }}

          .q-btn.bg-primary {{ font-family: 'Oswald', sans-serif; letter-spacing: 0.06em; text-transform: uppercase; }}

          @keyframes miFadeUp {{
            from {{ opacity: 0; transform: translateY(8px); }}
            to {{ opacity: 1; transform: translateY(0); }}
          }}
          @media (max-width: 960px) {{
            .mi-sidebar {{ width: 76px; }}
            .mi-sidebar .mi-nav-label,
            .mi-sidebar .mi-brand-sub,
            .mi-sidebar .mi-brand-line {{ display: none; }}
          }}
          .q-drawer {{ background: transparent !important; }}
          .q-field--dense .q-field__control {{ min-height: 38px; }}
        </style>
        """
    )
