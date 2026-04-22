TME Spatial Windows Icon

Important note
- Windows batch files (.bat) do not support an embedded custom icon directly in File Explorer.
- To show the TME Spatial icon, use the included shortcut instead of the raw .bat file.

Files included
- Launch_TME_Spatial.bat
- TME_Spatial.ico
- Create_TME_Spatial_Shortcut.ps1

How to create the icon shortcut on Windows
1. Open PowerShell in this folder.
2. Run:
   powershell -ExecutionPolicy Bypass -File .\Create_TME_Spatial_Shortcut.ps1
3. A new shortcut named Launch TME Spatial.lnk will appear in the folder.
4. Double-click that shortcut to launch the app with the custom icon.

Manual alternative
- Right-click Launch_TME_Spatial.bat and choose Create shortcut.
- Right-click the new shortcut and choose Properties.
- Open Change Icon.
- Select TME_Spatial.ico from this folder.

Packaging note
- Keep TME_Spatial.ico in the same folder when sharing the Windows app package.
- The shortcut uses that icon file.
