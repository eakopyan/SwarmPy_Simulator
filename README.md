# Before starting

## Install the virtual environment
Open the VS Code terminal and type:

`.\virtual_environment\Scripts\Activate.ps1`

This will launch the activation script through Powershell. If you get an execution error because the script execution is deactivated on your device, then:

1. Open a Powershell terminal as administrator
2. Type `set-executionpolicy unrestricted`
3. Re-launch your activation script.

All needed packages are listed in the `.\virtual_environment\packages.txt` file. To install them, type on the VS Code terminal:

`python.exe -m pip install -r .\virtual_environment\packages.txt`