"""Profiling integration helpers.

This module provides a small, dependency-light facade over three
complementary profiling backends so any example/benchmark can opt into
them via environment variables without hard-coding profiler plumbing:

  * JAX Profiler API (``jax.profiler``) — captures device/host traces in
    the Perfetto / TensorBoard ``trace_event`` format. The emitted
    ``*.trace.json.gz`` (or ``.pb``) under the log directory can be loaded
    directly in https://ui.perfetto.dev or TensorBoard's Trace Viewer.

  * Perfetto — JAX's profiler already writes Perfetto-compatible protobuf
    traces. We additionally surface the exact UI-load instructions and,
    when the standalone ``perfetto`` Python package is importable, expose
    its trace-processor for programmatic queries.

  * Scalene — a sampling CPU+GPU+memory profiler. Scalene profiles the
    whole process when the script is launched under ``scalene``; this
    module detects that context and annotates the run, and otherwise
    prints the exact command to re-launch under Scalene.

Environment variables (all optional):

  PROFILE              Master switch. Comma-separated subset of
                       ``jax,perfetto,scalene`` (or ``all``). Empty/unset
                       disables all profiling (zero overhead).
  PROFILE_DIR          Output directory for traces (default ``./profiles``).
  PROFILE_NAME         Basename prefix for emitted artifacts.

Typical usage in a benchmark::

    from qqn_jax.profiling import profile_session, profile_region

    with profile_session("mlp_comparison"):
        ...                       # whole-run device+host trace
        with profile_region("QQN-L80"):
            run_qqn(...)          # named sub-range in the Perfetto trace
"""

import contextlib
import os
import sys
import time


def _enabled_backends():
    """Parse ``PROFILE`` into the set of requested backends."""
    raw = os.environ.get("PROFILE", "").strip().lower()
    if raw in ("", "0", "false", "off", "none"):
        return frozenset()
    if raw in ("1", "true", "on", "all"):
        return frozenset({"jax", "perfetto", "scalene"})
    toks = {t.strip() for t in raw.split(",") if t.strip()}

    return frozenset(toks)


def _profile_dir():
    d = os.environ.get("PROFILE_DIR", "./profiles")
    os.makedirs(d, exist_ok=True)
    return d


def _profile_name(default):
    return os.environ.get("PROFILE_NAME", default)


def scalene_active():
    """Return True if the process is running under Scalene."""

    if os.environ.get("SCALENE_PROFILE") is not None:
        return True
    return "scalene" in sys.modules or any("scalene" in m for m in sys.modules)


def _print_scalene_hint(script_argv):
    """Print the command to re-run the current script under Scalene."""
    try:

        have = True
    except Exception:
        have = False
    cmd = "scalene " + " ".join(script_argv)
    if have:
        print(
            "[profile] Scalene is installed. To capture a CPU+GPU+memory "
            "profile, re-run under Scalene:\n"
            f"             {cmd}\n"
            "          (add --html --outfile profile.html for a report)."
        )
    else:
        print(
            "[profile] Scalene requested but not installed. Install with:\n"
            "             pip install scalene\n"
            f"          then run:  {cmd}"
        )


@contextlib.contextmanager
def profile_session(name="run"):
    """Context manager wrapping a whole benchmark run with the selected
    profilers.

    * For ``jax``/``perfetto``: starts ``jax.profiler.start_trace`` writing
      a Perfetto/TensorBoard trace into ``PROFILE_DIR`` and stops it on
      exit, then prints the exact UI-load instructions.
    * For ``scalene``: annotates whether the process is already under
      Scalene and, if not, prints the re-launch command (Scalene must wrap
      the *whole* interpreter, so it cannot be toggled mid-process).

    When no backends are enabled this is a zero-overhead no-op.
    """
    backends = _enabled_backends()
    if not backends:
        yield
        return

    outdir = _profile_dir()
    base = _profile_name(name)
    ts = time.strftime("%Y%m%d-%H%M%S")

    jax_trace = "jax" in backends or "perfetto" in backends
    started = False
    trace_dir = os.path.join(outdir, f"{base}_{ts}")

    if jax_trace:
        try:
            import jax

            os.makedirs(trace_dir, exist_ok=True)

            try:
                jax.profiler.start_trace(trace_dir, create_perfetto_trace=True)
            except TypeError:
                jax.profiler.start_trace(trace_dir)
            started = True
            print(f"[profile] JAX/Perfetto trace -> {trace_dir}")
        except Exception as exc:
            print(f"[profile] Failed to start JAX profiler: {exc}")

    if "scalene" in backends:
        if scalene_active():
            print("[profile] Running under Scalene (whole-process profile).")
        else:
            _print_scalene_hint([sys.executable, *sys.argv])

    try:
        yield
    finally:
        if started:
            try:
                import jax

                jax.profiler.stop_trace()
                print(
                    "[profile] Trace written. View it with EITHER:\n"
                    f"            * https://ui.perfetto.dev  (open "
                    f"{trace_dir}/**/*.perfetto_trace.json.gz)\n"
                    f"            * tensorboard --logdir {outdir}  "
                    "(Profile -> Trace Viewer)"
                )
            except Exception as exc:
                print(f"[profile] Failed to stop JAX profiler: {exc}")


@contextlib.contextmanager
def profile_region(label):
    """Annotate a named sub-range inside an active JAX/Perfetto trace.

    Uses ``jax.profiler.TraceAnnotation`` so the region shows up as a named
    span (e.g. one per optimizer variant) in the Perfetto timeline. A
    no-op when profiling is disabled or JAX is unavailable.
    """
    backends = _enabled_backends()
    if not ("jax" in backends or "perfetto" in backends):
        yield
        return
    try:
        import jax

        with jax.profiler.TraceAnnotation(label):
            yield
    except Exception:
        yield


def device_memory_report():
    """Return a short human-readable JAX device memory summary, or ``None``.

    Useful to log alongside Perfetto traces / Scalene reports so the
    memory-pressure context (relevant to the OOM-avoidance notes in the
    benchmarks) is captured next to the timing data.
    """
    try:
        import jax

        lines = []
        for dev in jax.devices():
            stats = getattr(dev, "memory_stats", None)
            if callable(stats):
                s = stats()
                if isinstance(s, dict):
                    in_use = s.get("bytes_in_use")
                    limit = s.get("bytes_limit")
                    if in_use is not None and limit:
                        lines.append(
                            f"{dev}: {in_use / 1e9:.2f} / {limit / 1e9:.2f} GiB"
                        )
        return "\n".join(lines) if lines else None
    except Exception:
        return None


__all__ = [
    "profile_session",
    "profile_region",
    "scalene_active",
    "device_memory_report",
]
