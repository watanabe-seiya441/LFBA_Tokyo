#include <Arduino.h>

// Pin configuration
const int LED_PINS[] = {2, 3, 4};
const int RELAY_PINS[] = {6, 7, 8, 9};
const int BUTTON_PINS[] = {A0, A1, A2, A3, 5};
const int TOGGLE_SWITCHES[] = {A4, A5};

// Global state variable (bit-based)
uint8_t state = 0;
uint8_t prev_toggle_state = 0;

void setup() {
    Serial.begin(9600);
    initializePins();

    // Initialize toggle switch state
    prev_toggle_state = readToggleSwitches();
    state |= prev_toggle_state;

    sendNotification('S');
}

void loop() {
    bool is_auto = state & (1 << 5);

    if (!is_auto) {
        updateStateFromButtons();
    }

    if (updateStateFromSwitches()) {
        sendNotification('S');
    }

    updateStateFromSerial(is_auto);
    updateOutputs();

    delay(100);
}

/**
 * Initializes all necessary pin modes.
 */
void initializePins() {
    for (int pin : LED_PINS) pinMode(pin, OUTPUT);
    for (int pin : RELAY_PINS) pinMode(pin, OUTPUT);
    for (int pin : BUTTON_PINS) pinMode(pin, INPUT_PULLUP);
    for (int pin : TOGGLE_SWITCHES) pinMode(pin, INPUT_PULLUP);
}

/**
 * Reads toggle switch states and returns the corresponding bitmask.
 */
uint8_t readToggleSwitches() {
    return (digitalRead(TOGGLE_SWITCHES[0]) << 5) | (digitalRead(TOGGLE_SWITCHES[1]) << 6);
}

/**
 * Handles button presses and updates the state.
 */
void updateStateFromButtons() {
    for (int i = 0; i < 5; i++) {
        if (debounce(BUTTON_PINS[i])) {
            state ^= (1 << i);
            sendNotification('S');
        }
    }
}

/**
 * Handles toggle switch changes. Returns true if state changed.
 */
bool updateStateFromSwitches() {
    uint8_t new_toggle_state = readToggleSwitches();
    if (new_toggle_state != prev_toggle_state) {
        state = (state & 0b00011111) | new_toggle_state;
        prev_toggle_state = new_toggle_state;
        return true;
    }
    return false;
}

/**
 * Reads serial input and updates the state if in auto mode.
 */
void updateStateFromSerial(bool is_auto) {
    if (Serial.available()) {
        String received = Serial.readStringUntil('\n');
        if (is_auto && received.startsWith("C") && received.length() == 6) {
            for (int i = 0; i < 4; i++) {
                if (received[i + 1] == '0' || received[i + 1] == '1') {
                    state = (state & ~(1 << i)) | ((received[i + 1] - '0') << i);
                }
            }
            sendNotification('S');
        }
    }
}

/**
 * Updates relays and LEDs based on the current state.
 */
void updateOutputs() {
    for (int i = 0; i < 4; i++) {
        digitalWrite(RELAY_PINS[i], state & (1 << i));
    }

    digitalWrite(LED_PINS[0], state & (1 << 5));
    digitalWrite(LED_PINS[1], state & (1 << 6));
    digitalWrite(LED_PINS[2], !(state & (1 << 6)));
}

/**
 * Simple debounce function for button inputs.
 */
bool debounce(int pin) {
    if (digitalRead(pin) == LOW) {
        delay(50); // Debounce delay
        if (digitalRead(pin) == LOW) {
            while (digitalRead(pin) == LOW); // Wait until release
            return true;
        }
    }
    return false;
}

/**
 * Sends a notification to the serial monitor.
 */
void sendNotification(char type) {
    Serial.print(type);
    for (int i = 0; i < 4; i++) Serial.print((state & (1 << i)) ? "1" : "0");
    Serial.print((state & (1 << 5)) ? "1" : "0");
    Serial.print((state & (1 << 6)) ? "1" : "0");
    Serial.println();
}
