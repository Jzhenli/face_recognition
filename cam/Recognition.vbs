Dim WshShell
Set WshShell = WScript.CreateObject("WScript.Shell")
WshShell.Run "Python\pythonw.exe CameraAPP.launch.pyw"
Set WshShell = Nothing