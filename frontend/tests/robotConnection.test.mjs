import test from 'node:test'
import assert from 'node:assert/strict'
import { RobotConnection } from '../src/services/robotConnection.ts'
import { typedCommand } from '../src/services/typedCommand.ts'

const settle = () => new Promise(resolve => setImmediate(resolve))
function fakeApi() {
  const api = {
    current: { running: false, target: null, status: 'idle', command_id: null, session_id: null, revision: 0, paused: false, instance_id: 'test-server' },
    commands: [], receipts: new Map(), loseReply: false,
    async status() { return { ...api.current } },
    async capabilities() { return { voiceMode: 'clip', speechProvider: 'fake', maxRecordingSeconds: 10, stopWord: 'stop', resumeWord: 'resume' } },
    async heartbeat() {},
    async direct(command) {
      api.commands.push(command)
      api.current = { ...api.current, running: true, status: 'moving', target: command.target, command_id: command.commandId, session_id: command.sessionId, revision: api.current.revision + 1, paused: false }
      api.receipts.set(command.commandId, { ...api.current })
      if (api.loseReply) throw new Error('response lost')
    },
    async receipt(id) { if (!api.receipts.has(id)) throw new Error('unknown'); return api.receipts.get(id) },
    async stop() { api.current = { ...api.current, running: false, paused: true, status: 'stopped', revision: api.current.revision + 1 } },
    async resume() { api.current = { ...api.current, paused: false, status: 'idle', revision: api.current.revision + 1 } },
    finish(status = 'arrived') { api.current = { ...api.current, running: false, status, paused: status !== 'arrived' }; api.receipts.set(api.current.command_id, { ...api.current }) },
  }
  return api
}
const voice = (target, intent = 'navigate') => ({ requestId: crypto.randomUUID(), transcript: target, intent, target, reason: null, message: null, simulated: true })
async function setup(t) {
  const api = fakeApi()
  const connection = new RobotConnection(api)
  const stop = connection.start()
  t.after(stop)
  await settle()
  return { api, connection }
}

test('queues tasks and advances only after arrival; duplicate interpretation is ignored', async t => {
  const { api, connection } = await setup(t)
  const first = voice('blue flag')
  await connection.accept(first)
  await connection.accept(first)
  await connection.accept(voice('red ball'))
  assert.equal(api.commands.length, 1)
  assert.equal(connection.getSnapshot().queue.length, 1)
  api.finish()
  await connection.refresh()
  assert.equal(api.commands.length, 2)
  assert.equal(api.commands[1].target, 'red ball')
})

test('failed navigation pauses and retains the queue; resume retries interrupted target', async t => {
  const { api, connection } = await setup(t)
  await connection.accept(voice('blue flag'))
  await connection.accept(voice('red ball'))
  api.finish('target_lost')
  await connection.refresh()
  assert.equal(connection.getSnapshot().paused, true)
  assert.equal(api.commands.length, 1)
  await connection.accept(voice('resume', 'resume'))
  assert.equal(api.commands.length, 2)
  assert.equal(api.commands[1].target, 'blue flag')
  assert.notEqual(api.commands[1].commandId, api.commands[0].commandId)
  assert.equal(connection.getSnapshot().queue[0].target, 'red ball')
})

test('lost direct response is reconciled by receipt without resubmitting movement', async t => {
  const { api, connection } = await setup(t)
  api.loseReply = true
  await connection.accept(voice('blue flag'))
  assert.match(connection.getSnapshot().message, /Outcome unknown/)
  await connection.refresh()
  assert.equal(api.commands.length, 1)
  assert.equal(connection.getSnapshot().paused, true)
  assert.match(connection.getSnapshot().message, /receipt recovered/)
})

test('stop retains active/pending tasks; reconnect never auto resumes', async t => {
  const { api, connection } = await setup(t)
  await connection.accept(voice('blue flag'))
  await connection.accept(voice('red ball'))
  await connection.accept(voice('stop', 'stop'))
  assert.equal(connection.getSnapshot().active.target, 'blue flag')
  assert.equal(connection.getSnapshot().queue.length, 1)
  const getStatus = api.status
  api.status = async () => { throw new Error('offline') }
  await connection.refresh()
  assert.equal(connection.getSnapshot().connected, false)
  api.status = getStatus
  await connection.refresh()
  assert.equal(connection.getSnapshot().paused, true)
  assert.equal(api.commands.length, 1)
})

test('a typed target dispatches through the same path as a spoken one', async t => {
  const { api, connection } = await setup(t)
  await connection.accept(typedCommand('the red ball'))
  assert.equal(api.commands.length, 1)
  assert.equal(api.commands[0].target, 'the red ball')
  // And inherits the queueing rather than racing the active command.
  await connection.accept(typedCommand('the blue flag'))
  assert.equal(api.commands.length, 1, 'the second waits for arrival')
  assert.equal(connection.getSnapshot().queue.length, 1)
  api.finish()
  await connection.refresh()
  assert.equal(api.commands[1].target, 'the blue flag')
})
