# Switch Control Board

The control board receives signals from the Central Manager and directly controls various equipment. This directory contains the Arduino file `switcher.ino` required to operate the control board.

## Installation
To use the control board, you need to write the Arduino file to the board.

1. **Write the Arduino File**: Use the Arduino IDE to write the `switcher.ino` file to the Arduino on the board.
2. **Connect to Central Manager**: Connect the Arduino to the Central Manager using a USB cable.
3. **Power the Board**: Connect the second USB port on the board to a power source using a USB cable. This could be a wall outlet or any USB power source.
4. **Power Confirmation**: If at least one of the lights is illuminated, the board is receiving power correctly.

## Board Features
The board includes 4 buttons, 2 toggle switches, and 3 indicator lights.

### Buttons
- Each button is connected to and controls a specific piece of equipment, such as lighting or fans.
- When a button is illuminated, the corresponding equipment is ON. When it is not illuminated, the equipment is OFF.

### Toggle Switches
- **Left Toggle Switch**: Switches between "Automation Mode" and "Training Mode".
  - **Up Position**: Automation Mode - Equipment is controlled automatically using the model.
  - **Down Position**: Training Mode - Equipment can be controlled manually.
- **Right Toggle Switch**: Controls the camera's power.
  - **Up Position**: Camera is ON.
  - **Down Position**: Camera is OFF.

### Lights
- The board has three indicator lights to provide visual feedback on the system status. (Detailed information about each light's function can be added as needed.)

