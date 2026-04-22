TME Spatial macOS Launcher

Files included
- TME Spatial.app
- launch_tme_spatial_macos.sh
- app.py
- requirements.txt

How to use
1. Open the folder.
2. Double-click TME Spatial.app.
3. A Terminal window will open and show setup progress.
4. The app should open in your default browser at http://localhost:8501

What the launcher does
- Checks whether Conda is installed.
- If Conda is available, it creates or reuses the environment named TME_spatial.
- If Conda is not available, it creates or reuses a local Python virtual environment.
- Installs or updates the required packages from requirements.txt.
- Starts the Streamlit app.

Python note
- If Python is missing and Homebrew is available, the launcher will try to install Python 3.11 automatically.
- If Python is missing and Homebrew is not installed, a message will appear telling the user what to install.

How to share this app
- Keep TME Spatial.app in the same folder as app.py, requirements.txt, and launch_tme_spatial_macos.sh.
- Zip the whole folder before sending it to another Mac user.
- After extracting, they can double-click TME Spatial.app.

If macOS blocks the app
- Right-click TME Spatial.app and choose Open.
- Then click Open again if macOS asks for confirmation.

If the browser does not open automatically
- Check the Terminal window for the local URL.
- Open the URL manually in a browser, usually http://localhost:8501
