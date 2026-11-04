// Light / dark theme (issue #1). "system" follows the operating system setting.

export type ThemeChoice = "system" | "light" | "dark";

const KEY = "dropchess.theme";
const ORDER: ThemeChoice[] = ["system", "light", "dark"];

function read(): ThemeChoice {
  try {
    const v = localStorage.getItem(KEY);
    return v === "light" || v === "dark" ? v : "system";
  } catch {
    return "system";
  }
}

let choice: ThemeChoice = read();

/** The theme actually showing right now. */
export function effectiveTheme(): "light" | "dark" {
  if (choice !== "system") return choice;
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function themeChoice(): ThemeChoice {
  return choice;
}

export function applyTheme() {
  const root = document.documentElement;
  if (choice === "system") delete root.dataset.theme;
  else root.dataset.theme = choice;
  const meta = document.querySelector('meta[name="theme-color"]');
  meta?.setAttribute("content", effectiveTheme() === "dark" ? "#151412" : "#f6f4ef");
}

/** Cycles system → light → dark and returns the new choice. */
export function cycleTheme(): ThemeChoice {
  return setTheme(ORDER[(ORDER.indexOf(choice) + 1) % ORDER.length]);
}

/** Switches straight to the opposite of what is showing now (the top-bar button). */
export function toggleTheme(): ThemeChoice {
  return setTheme(effectiveTheme() === "dark" ? "light" : "dark");
}

export function setTheme(next: ThemeChoice): ThemeChoice {
  choice = next;
  try {
    if (choice === "system") localStorage.removeItem(KEY);
    else localStorage.setItem(KEY, choice);
  } catch {
    /* storage unavailable: the choice lasts for this page only */
  }
  applyTheme();
  return choice;
}

export function themeLabel(c: ThemeChoice = choice): string {
  return c === "system" ? "Theme: follow system" : c === "light" ? "Theme: light" : "Theme: dark";
}

// Keep the theme-color meta tag in sync when the system setting changes.
window.matchMedia?.("(prefers-color-scheme: dark)").addEventListener?.("change", applyTheme);
applyTheme();
