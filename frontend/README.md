# Sparkle frontend

A monochrome mock operator dashboard for the RC car.

```sh
npm install
npm run dev
```

Open the localhost URL printed by Vite. Mock mode is enabled by default. This layout preview does not connect to a robot or request microphone permission.

- Edit the demo transcript and press the microphone twice to simulate a command.
- Enter `stop` while mock listening to pause navigation and preserve the queue.
- Enter `resume` and process it to continue.
- Use **Complete target** to simulate arrival and advance the queue.
- Use **Simulate disconnect** to inspect the lost-connection screen. Restoring the connection retains the paused queue.
- Refresh to reset the mock scenario.

`VITE_STOP_WORD` configures the stop keyword. Set `VITE_MOCK_MODE=true` explicitly if desired. Real speech and robot HTTP integration are not implemented in this layout pass.

Validation: `npm run build` and `npm run lint`.
