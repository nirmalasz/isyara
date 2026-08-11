from collections import deque


class SequenceBuffer:
    def __init__(self, max_length):
        self.max_length = max_length
        self.frames = deque(maxlen=max_length)

    def append(self, frame):
        self.frames.append(frame)

    def ready(self):
        return len(self.frames) == self.max_length

    def as_list(self):
        return list(self.frames)

    def clear(self):
        self.frames.clear()
