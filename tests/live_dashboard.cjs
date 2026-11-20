const assert = require('node:assert/strict');
const fs = require('node:fs');
const test = require('node:test');
const vm = require('node:vm');

const script = fs.readFileSync(0, 'utf8').split('<script>')[1].split('</script>')[0];
const flush = () => new Promise(resolve => setImmediate(resolve));

async function dashboard() {
    const elements = new Map();
    const timers = new Map();
    const ui = id => {
        if (!elements.has(id)) elements.set(id, {textContent: '', value: '', hidden: true});
        return elements.get(id);
    };
    let state = () => ({ok: true, json: async () => ({status: 'idle'})});
    let send = () => ({ok: true});
    let sends = 0;
    vm.runInNewContext(script + '\ndraw = () => {};', {
        document: {getElementById: ui, addEventListener() {}, body: {classList: {toggle() {}}}},
        AbortController,
        IntersectionObserver: class { observe() {} },
        setTimeout(callback, delay) {
            timers.set(callback, delay);
            return callback;
        },
        clearTimeout(callback) {
            timers.delete(callback);
        },
        async fetch(path, options) {
            if (path === '/state') return state(options);
            assert.equal(path, '/instruct');
            sends++;
            return send(options);
        },
    });
    await flush();
    return {
        ui,
        submit: () => ui('form').onsubmit({preventDefault() {}}),
        state: handler => { state = handler; },
        send: handler => { send = handler; },
        sends: () => sends,
        async tick(delay) {
            const timer = [...timers].find(([, ms]) => ms === delay);
            assert.ok(timer, `No timer scheduled for ${delay} ms`);
            timers.delete(timer[0]);
            timer[0]();
            await flush();
        },
    };
}

const networkFailure = () => { throw new TypeError('Failed to fetch'); };
const draft = 'Look at the table';

test('an unreachable server keeps the draft and labels the last state as stale', async () => {
    const page = await dashboard();
    assert.equal(page.ui('status').textContent, 'idle');
    page.state(networkFailure);
    page.send(networkFailure);
    await page.tick(350);
    page.ui('cmd').value = draft;
    await page.submit();
    assert.equal(page.ui('cmd').value, draft);
    assert.equal(page.ui('status').textContent, 'Live updates unavailable');
    assert.equal(page.ui('state-stale').hidden, false);
    assert.equal(page.ui('instruction').textContent, 'Tell the robot what to do');
    assert.match(page.ui('send-error').textContent, /confirm.*sent/i);

    page.state(() => ({ok: true, json: async () => ({status: 'running', instruction: 'Look around'})}));
    await page.tick(350);
    assert.equal(page.ui('status').textContent, 'running');
    assert.equal(page.ui('state-stale').hidden, true);
    assert.match(page.ui('send-error').textContent, /confirm.*sent/i);
    assert.equal(page.ui('cmd').value, draft);
});

test('an HTTP failure keeps the draft; only a successful retry clears it', async () => {
    const page = await dashboard();
    page.state(() => ({ok: false, status: 503}));
    await page.tick(350);
    assert.equal(page.ui('status').textContent, 'Live updates unavailable');
    let reply;
    page.send(() => new Promise(resolve => { reply = resolve; }));
    page.ui('cmd').value = draft;
    const pending = page.submit();
    assert.equal(page.ui('cmd').value, draft);
    assert.equal(page.ui('send').disabled, true);
    await page.submit();
    assert.equal(page.sends(), 1);
    reply({ok: false, status: 503});
    await pending;
    assert.equal(page.ui('cmd').value, draft);
    assert.equal(page.ui('send').disabled, false);

    page.send(({body}) => {
        assert.equal(JSON.parse(body).text, draft);
        return {ok: true};
    });
    await page.submit();
    assert.equal(page.ui('cmd').value, '');
    assert.equal(page.ui('send-error').textContent, '');
    assert.equal(page.ui('status').textContent, 'Live updates unavailable');
});

test('a successful send does not erase edits made while waiting for the server', async () => {
    const page = await dashboard();
    let reply;
    page.send(() => new Promise(resolve => { reply = resolve; }));
    page.ui('cmd').value = draft;
    const pending = page.submit();
    page.ui('cmd').value = 'A new draft';
    reply({ok: true});
    await pending;
    assert.equal(page.ui('cmd').value, 'A new draft');
    assert.equal(page.ui('status').textContent, 'idle');
});

test('a stalled send times out and leaves the draft available', async () => {
    const page = await dashboard();
    page.send(({signal}) => new Promise((_, reject) => {
        signal.addEventListener('abort', () => reject(signal.reason), {once: true});
    }));
    page.ui('cmd').value = draft;
    const pending = page.submit();
    await page.tick(5000);
    await pending;
    assert.equal(page.ui('cmd').value, draft);
    assert.equal(page.ui('send').disabled, false);
    assert.match(page.ui('send-error').textContent, /confirm.*sent/i);
});

test('a suite case disables every command control and explains interactive mode', async () => {
    const page = await dashboard();
    page.state(() => ({ok: true, json: async () => ({read_only: true, scene: 'O3', scenes: ['O3']})}));
    await page.tick(350);
    assert.equal(page.ui('read-only').hidden, false);
    assert.match(page.ui('read-only').textContent, /Watching suite case O3.*read-only.*without --case/);
    for (const id of ['cmd', 'send', 'stop', 'scene', 'talk']) assert.equal(page.ui(id).disabled, true, id);
    page.ui('cmd').value = draft;
    await page.submit();
    assert.equal(page.sends(), 0);
    page.state(() => ({ok: true, json: async () => ({read_only: false})}));
    await page.tick(350);
    assert.equal(page.ui('read-only').hidden, true);
    for (const id of ['cmd', 'send', 'stop', 'scene', 'talk']) assert.equal(page.ui(id).disabled, false, id);
});

test('a read-only refusal explains why and preserves the draft', async () => {
    const page = await dashboard();
    const error = 'This page only watches one suite case. Run the demo without --case to give commands.';
    page.send(() => ({ok: false, status: 409, json: async () => ({ok: false, error})}));
    page.ui('cmd').value = draft;
    await page.submit();
    assert.equal(page.ui('cmd').value, draft);
    assert.equal(page.ui('send-error').textContent, error);
});
