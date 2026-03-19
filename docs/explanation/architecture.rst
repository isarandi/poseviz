Architecture
============

PoseViz separates visualization from user code by running the renderer in a
dedicated process. This page explains why that separation exists and how data
flows between processes.


A natural way to write pose estimation code for a video or webcam stream is a simple loop::

    for frame in video:
        poses = model.predict(frame)
        viz.show(frame, poses)

But OpenGL (and GUI toolkits generally) wants to own the event loop. The
renderer needs to poll for window events, handle redraws, and run at a
consistent frame rate. This requires writing the whole inference loop in a different way. The design goal of PoseViz is to be able to easily add visualization to existing inference code (which is not mainly about visualization) without having to restructure it.

The solution is process separation. User code runs in the main process with
its natural control flow. The renderer runs in a child process with its own
event loop. A message queue connects them.

But frames are large, so sending the frame data in a queue is expensive. A 1920x1080 RGB image is about 6 MB, so this would add latency.
At 30 fps, it's 180 MB/s of copying overhead. We therefore use shared memory
to hold frame data, and only send small messages with metadata through the
queue. This avoids copies and keeps latency low.

Process model
-------------

PoseViz uses three execution contexts:

1. **Main process**: Runs user code. Receives frames, dispatches preprocessing
   to a thread pool, queues messages for the visualizer.

2. **Waiter thread**: A daemon thread in the main process. Waits for async
   preprocessing results and forwards complete messages to the visualizer.

3. **Visualizer process**: A spawned child process. Runs the OpenGL event loop,
   receives messages, reconstructs frames from shared memory, renders.

The data flow::

    ┌──────────────────────────────┐
    │         Main process         │
    │                              │
    │         User code            │
    │             │                │
    │             ▼                │
    │        q_undistort           │
    │             │                │
    │             ▼                │       ┌──────────────────┐
    │    ┌─────────────────┐       │       │ Shared memory    │
    │    │ Undistortion    │───────┼──────►│ ring buffer      │
    │    │ thread pool     │       │       └────────┬─────────┘
    │    └─────────────────┘       │                │
    │             │                │                │
    │             ▼                │                │
    │       q_messages_pre         │                │
    │             │                │                │
    │             ▼                │                │
    │      ┌──────────────┐        │                │
    │      │ Waiter thread│        │                │
    │      └──────┬───────┘        │                │
    │             │                │                │
    └─────────────┼────────────────┘                │
                  │                                 │
                  ▼                                 │
            q_messages_post                         │
                  │                                 │
                  ▼                                 │
    ┌──────────────────────────────┐                │
    │      Visualizer process      │◄───────────────┘
    │                              │  (reads frame data)
    └──────────────────────────────┘

Undistortion thread pool
~~~~~~~~~~~~~~~~~~~~~~~~

Undistorting and resizing frames is CPU-intensive. To keep up with real-time
video, PoseViz uses a thread pool to process multiple frames in parallel, while the main user thread can move on to running inference for the next frames.

The thread pool may be finished with the undistortion of frames in a different
order than they were submitted. To preserve message order, a dedicated waiter
thread blocks on the async results and forwards completed messages in order. There are two message queues:

- ``q_messages_pre``: Main thread enqueues messages with async result handles. Waiter thread waits on results.
- ``q_messages_post``: Waiter thread enqueues completed messages. Visualizer process dequeues.

When the main thread calls ``update()``, it dispatches undistortion work and
immediately queues a message containing the async result handle into
``q_messages_pre``. The waiter thread continuously reads from this queue. For
each message, the waiter
thread blocks on ``result.get()``, then forwards the complete message to the
visualizer process via ``q_messages_post``.
This keeps the main thread responsive while ensuring messages
arrive in order with their data ready.

The ring buffer
---------------

We could also put frame data in the ``q_messages_post``. However, this would require copying the frame between processes. Shared memory avoids this copy. But shared memory
requires coordination: the producer must not overwrite data the consumer is
still reading.

PoseViz uses a ring buffer for this. The buffer has N slots, where N equals the sum of
all queue capacities plus one::

    N = queue_size_undist + queue_size_waiter + queue_size_post + 1

Each slot holds one frame's worth of pixels. The main process writes to slot
``i``, then increments ``i`` modulo N. The message sent to the visualizer
includes the slot index, so the visualizer knows where to read.

The sizing ensures safety: by the time slot ``i`` is reused, the visualizer
has finished with the frame that was stored there. The ``+1`` accounts for
the frame currently being processed.

Slot allocation happens inside ``update_multiview()``. Each view's frame is
written to ``self.ring_index``, and the index advances after all views are
processed::

    # (inside the per-view loop)
    dst = self.frame_rings[i].get_slot(self.ring_index, ...)

    # (after the loop)
    self.ring_index = (self.ring_index + 1) % self.ringbuffer_size

If the frame is dropped (queue full, ``block=False``), the method returns
early before the index advances, so no slot is wasted.

Message passing
---------------

Messages are Python dataclasses sent through ``multiprocessing.JoinableQueue``.
The ``JoinableQueue`` supports ``task_done()`` calls, which enable the sender
to wait until the receiver has processed all pending messages.

Message types:

- ``UpdateScene``: New frame data. Contains ``view_infos`` (camera, boxes,
  poses, frame metadata) and optional override camera for the view.

- ``StartSequence``: Begin recording video output. Carries path, fps, audio.

- ``EndSequence``: Finalize current video file.

- ``Pause`` / ``Resume``: Toggle visualization pause state.

- ``ReinitCameraView``: Signal that a new sequence is starting; reset view
  camera if configured to snap on scene change.

- ``Quit``: Shut down the visualizer process.

The ``UpdateScene`` message does not contain the frame pixels directly.
Instead, it contains ``(shape, dtype, ring_index)`` tuples. The visualizer
reconstructs the ``np.ndarray`` by indexing into the shared memory buffer.

Backpressure works through queue capacity. When the queue is full:

- If ``block=True`` (default): ``update()`` blocks until space is available. This
  ensures every frame is rendered, at the cost of latency if the visualizer
  cannot keep up. This is suitable for processing video files.
- If ``block=False``: ``update()`` returns immediately, dropping the frame. This is useful for real-time webcam visualization, where latency is more important than
  rendering every frame. However, PoseViz is faster than real-time (typically 70-100+ fps), so frame drops should be very rare.

The ``task_done()`` protocol allows ``close()`` to wait until all queued
messages are processed before terminating the visualizer.

Frame data lifecycle
--------------------

A frame's journey from user code to rendered pixels:

1. **User provides**: ``np.ndarray`` with shape ``(H, W, 3)``, dtype ``uint8``.

2. **Main process**: Allocates a destination buffer in the ring buffer at the
   current slot index. Dispatches ``downscale_and_undistort_view_info`` to the
   thread pool (more precisely, its internal input queue), passing the source frame and destination buffer.

3. **Thread pool**: Downscales the frame (if configured), applies lens
   undistortion using the camera model, writes the result to the shared memory
   buffer. Returns ``(shape, dtype, slot_index)`` instead of the array itself.

4. **Waiter thread**: Calls ``result.get()`` on the async handle, blocking
   until undistortion completes. The returned ``ViewInfo`` now has its
   ``frame`` field set to the metadata tuple. Forwards the message to
   ``q_messages_post``.

5. **Visualizer process**: Receives the message, extracts
   ``(shape, dtype, index)`` from each ``ViewInfo.frame``. Calls
   ``np_from_raw_array()`` to wrap the shared memory region as an
   ``np.ndarray``::

       np.frombuffer(
           buffer=raw_array,
           dtype=dtype,
           count=np.prod(shape),
           offset=index * np.prod(shape) * itemsize,
       ).reshape(shape)

6. **Rendering**: The reconstructed array is uploaded to a GL texture.
   The texture is mapped onto a quad positioned at the camera's image plane.
   Poses are rendered as spheres and tubes. The scene is drawn to an
   offscreen framebuffer, then blitted to the window (and/or encoded to video).

The indirection through ``(shape, dtype, index)`` is necessary because
``np.ndarray`` objects cannot be pickled with shared memory backing. We
serialize the metadata and reconstruct on the other side.
