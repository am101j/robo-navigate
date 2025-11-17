# How to Find or Create Your ROS2 Workspace

## Quick Check: Do You Have ROS2 Installed?

```bash
# Check if ROS2 is installed
ros2 --help
```

If this works, ROS2 is installed. If not, you need to install ROS2 first.

## Finding Your ROS2 Workspace

### Method 1: Check Common Locations

**On Linux:**
```bash
# Check home directory
ls ~/ros2_ws
ls ~/ros_ws
ls ~/workspace

# Check if sourced in your shell
echo $ROS_WORKSPACE
```

**On Windows (WSL or native):**
```powershell
# Check common locations
Test-Path "$env:USERPROFILE\ros2_ws"
Test-Path "C:\dev\ros2_ws"
Test-Path "C:\ros2_ws"
```

### Method 2: Search for Workspace

**Linux:**
```bash
# Search for directories named "ros2_ws" or "ros_ws"
find ~ -type d -name "*ros*ws" 2>/dev/null
```

**Windows PowerShell:**
```powershell
# Search for ROS2 workspace directories
Get-ChildItem -Path $env:USERPROFILE -Recurse -Directory -Filter "*ros*ws" -ErrorAction SilentlyContinue -Depth 2 | Select-Object FullName
```

### Method 3: Check Your Current Directory

You might already be in a workspace! Check for these files:
```bash
# Look for these files/directories:
ls src/          # Should contain ROS2 packages
ls install/      # Build output
ls build/        # Build cache
ls log/          # Build logs
```

If you see these, you're already in a workspace!

## Creating a New ROS2 Workspace

If you don't have a workspace, create one:

### On Linux:

```bash
# Create workspace directory
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Initialize workspace (if needed)
# Workspace is ready when you have src/ directory
```

### On Windows:

```powershell
# Create workspace directory
New-Item -ItemType Directory -Path "$env:USERPROFILE\ros2_ws\src" -Force
cd "$env:USERPROFILE\ros2_ws"
```

## Using Your Current Directory as Workspace

**You can use the current `Robot` directory as your workspace!**

Since you already have the `robot_gazebo` package here, you can:

### Option 1: Create workspace structure here

```powershell
# In your current Robot directory
# Create workspace structure
New-Item -ItemType Directory -Path "src" -Force
New-Item -ItemType Directory -Path "install" -Force
New-Item -ItemType Directory -Path "build" -Force
New-Item -ItemType Directory -Path "log" -Force

# Move robot_gazebo to src/
Move-Item robot_gazebo src/
```

### Option 2: Use current directory directly

If `robot_gazebo` is already here, you can build it directly:

```powershell
# Make sure you're in the directory containing robot_gazebo
cd C:\Users\maksa\Projects\Robot

# Create src directory and move package
New-Item -ItemType Directory -Path "src" -Force
Move-Item robot_gazebo src\robot_gazebo

# Now build
colcon build --packages-select robot_gazebo
```

## Standard ROS2 Workspace Structure

A ROS2 workspace should look like this:

```
ros2_ws/                    # Your workspace root
├── src/                    # Source code (packages go here)
│   └── robot_gazebo/       # Your package
├── install/                # Built packages (created by colcon)
├── build/                  # Build cache (created by colcon)
└── log/                    # Build logs (created by colcon)
```

## Quick Setup Script

Here's a script to set up everything:

**Linux:**
```bash
#!/bin/bash
# Create workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Copy your package (adjust path)
cp -r /path/to/robot_gazebo src/

# Build
colcon build --packages-select robot_gazebo
source install/setup.bash

echo "Workspace ready at: $(pwd)"
```

**Windows PowerShell:**
```powershell
# Create workspace
$workspace = "$env:USERPROFILE\ros2_ws"
New-Item -ItemType Directory -Path "$workspace\src" -Force
cd $workspace

# Copy your package (adjust path)
Copy-Item "C:\Users\maksa\Projects\Robot\robot_gazebo" -Destination "src\" -Recurse

# Build
colcon build --packages-select robot_gazebo
. install\setup.bash

Write-Host "Workspace ready at: $workspace"
```

## Verify Your Workspace

After setting up, verify it works:

```bash
# Check package is found
ros2 pkg list | grep robot_gazebo

# Check executables are available
ros2 pkg executables robot_gazebo

# Should list all your nodes
```

## Common Workspace Locations

- **Linux**: `~/ros2_ws`, `~/ros_ws`, `~/workspace`
- **Windows**: `C:\dev\ros2_ws`, `%USERPROFILE%\ros2_ws`
- **WSL**: Same as Linux (`~/ros2_ws`)

## Next Steps

Once you have your workspace:

1. **Build the package:**
   ```bash
   colcon build --packages-select robot_gazebo
   ```

2. **Source the workspace:**
   ```bash
   source install/setup.bash  # Linux/WSL
   # or
   . install\setup.bash       # Windows PowerShell
   ```

3. **Test it:**
   ```bash
   ros2 launch robot_gazebo spawn_robot.launch.py
   ```



