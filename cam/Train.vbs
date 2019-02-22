Dim WshShell
Set WshShell = WScript.CreateObject("WScript.Shell")
WshShell.Run "Python\python.exe train.launch.pyw"
Set WshShell = Nothing