"""The visualizer process must terminate when its parent dies, on every OS.

Linux uses prctl(PR_SET_PDEATHSIG); elsewhere (macOS) a watchdog thread notices the
re-parenting. On Linux both paths are exercised, the macOS one by pretending to be on
macOS (sys.platform, and no loadable libc.so.6); elsewhere the native path runs. A middle process starts the child (which arms the
mechanism and then sleeps), then dies abruptly with os._exit; the child must be gone
within a few seconds.

Run: python tests/parent_death_test.py   (exits 0 on success)
"""
import multiprocessing as mp
import os
import subprocess
import sys
import time


def child(platform, pid_queue):
    from poseviz.poseviz import _terminate_on_parent_death

    if platform != sys.platform and platform == 'darwin':
        simulate_macos()
    _terminate_on_parent_death()
    pid_queue.put(os.getpid())
    time.sleep(60)


def simulate_macos():
    """Pretend to be on macOS on another Unix: the platform name, and no libc.so.6."""
    import ctypes

    sys.platform = 'darwin'  # after the imports, which read the real platform
    real_cdll = ctypes.CDLL

    def cdll(name, *args, **kwargs):
        if 'libc.so' in str(name):
            raise OSError(f'{name}: cannot open shared object file')
        return real_cdll(name, *args, **kwargs)

    ctypes.CDLL = cdll


def parent(platform, pid_queue):
    ctx = mp.get_context('spawn')
    ctx.Process(target=child, args=(platform, pid_queue)).start()
    time.sleep(3)  # let the child arm its watch
    os._exit(0)  # die without cleanup, as a crashed producer would


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie (exited, not yet reaped by its new parent) counts as gone
    state = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)], capture_output=True, text=True)
    state = state.stdout.strip()
    return bool(state) and not state.startswith('Z')  # empty: gone in the meantime


def run(platform):
    ctx = mp.get_context('spawn')
    pid_queue = ctx.Queue()
    p = ctx.Process(target=parent, args=(platform, pid_queue))
    p.start()
    try:
        child_pid = pid_queue.get(timeout=30)
    except Exception:
        p.join()
        raise SystemExit(f'{platform}: the child failed while arming parent-death handling')
    p.join()
    deadline = time.monotonic() + 5
    while alive(child_pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    if alive(child_pid):
        try:
            os.kill(child_pid, 9)
        except ProcessLookupError:
            pass
        raise SystemExit(f'{platform}: child {child_pid} survived its parent')
    print(f'{platform}: child terminated after parent death')


if __name__ == '__main__':
    # The real platform's path, plus the macOS path simulated on Linux (the Linux path
    # needs a real Linux kernel and cannot be simulated elsewhere)
    platforms = ['linux', 'darwin'] if sys.platform.startswith('linux') else [sys.platform]
    for platform in platforms:
        run(platform)
    print('PARENT-DEATH TEST PASSED')
