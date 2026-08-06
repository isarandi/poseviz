class VisualizerError(RuntimeError):
    """An error from the visualizer process, re-raised in the producer process.

    Raised by PoseViz API calls (update, new_sequence_output,
    finalize_sequence_output, close) when the visualizer reports that a
    requested output (video, camera trajectory) cannot be produced, or that the
    visualizer process itself failed. The message contains the original error
    and the visualizer-side traceback.
    """
