from cam.display import DisplayVideo


def run():
    print("waiting for system init... ")
    DisplayVideo(src=0).display()
    print("system exit.")