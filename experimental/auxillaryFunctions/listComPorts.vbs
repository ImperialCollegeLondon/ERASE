Set portList = GetComPorts()

portnames = portList.Keys
for each pname in portnames
    Set portinfo = portList.item(pname)
    wscript.echo pname & " - " & _
           portinfo.Manufacturer & " - " & _
           portinfo.PNPDeviceID & " - " & _
           portinfo.Name
Next

Function GetComPorts()
    set portList = CreateObject("Scripting.Dictionary")
    strComputer = "."
    set objWMIService = GetObject("winmgmts:\\" & strComputer & "\root\cimv2")
    set colItems = objWMIService.ExecQuery ("Select * from Win32_PnPEntity")
    for each objItem in colItems
        If Not IsNull(objItem.Name) Then
               set objRgx = CreateObject("vbScript.RegExp")
               objRgx.Pattern = "COM[0-9]+"
                        Set objRegMatches = objRgx.Execute(objItem.Name)
                        if objRegMatches.Count = 1 Then  portList.Add objRegMatches.Item(0).Value, objItem
                    End if
    Next
    set GetComPorts = portList
End Function