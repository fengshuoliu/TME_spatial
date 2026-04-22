TME Spatial Windows Launcher

Files in this folder:
- Launch_TME_Spatial.bat
- launch_tme_spatial.ps1
- TME_Spatial.ico
- Create_TME_Spatial_Shortcut.ps1
- app.py
- requirements.txt

How to start the app on Windows:

1. Extract or copy this whole folder to your Windows computer.
2. Open the folder.
3. Double-click Launch_TME_Spatial.bat.
4. Wait while the launcher checks your system.

Custom icon:
- Windows does not allow a `.bat` file itself to carry a custom Explorer icon.
- A matching TME Spatial icon is included as `TME_Spatial.ico`.
- To use it, run `Create_TME_Spatial_Shortcut.ps1` once to create `Launch TME Spatial.lnk`.
- After that, launch the app from the shortcut if you want the custom icon in Explorer.

What the launcher does automatically:
- If Conda is installed, it creates or reuses an environment named TME_spatial.
- If Conda is not installed, it uses Python directly.
- If Python is missing, it first tries `winget` and then falls back to the official Python installer from python.org.
- It installs all required packages from requirements.txt.
- It runs: python -m streamlit run app.py
- Streamlit should then open in your default web browser.

Important notes:
- The first launch may take a while because Python packages need to be installed.
- Please keep all files together in the same folder.
- Internet access is needed the first time so packages can be downloaded.
- If Windows asks for permission to run PowerShell or install software, allow it.

If the browser does not open automatically:
- Look at the PowerShell window.
- Copy the local Streamlit address, usually something like:
  http://localhost:8501
- Paste it into your browser.

If the launcher fails:
- Keep the PowerShell window open.
- Read the error message shown there.
- A diagnostic log is also written to `launcher_log.txt` in the same folder.
- Common fixes:
  - Re-run Launch_TME_Spatial.bat as Administrator.
  - Make sure you are connected to the internet.
  - Update Windows PowerShell.
  - Install Python 3.11 manually from python.org, then run the launcher again.
  - Install Miniconda manually if your IT policy blocks Python installation.

Manual fallback:
- If needed, you can still follow the manual PowerShell installation steps provided by the app author.
