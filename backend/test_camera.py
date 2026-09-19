from camera import FakeCamera
from scenarios import FakeScene


def test_frame_no_increments():
    cam = FakeCamera(FakeScene(boxes=[(300, 300, 700, 700)]))
    assert cam.frame_no == 0
    cam.read()
    cam.read()
    cam.read()
    assert cam.frame_no == 3