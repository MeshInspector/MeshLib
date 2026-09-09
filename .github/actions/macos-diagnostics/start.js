const { execFileSync, spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

execFileSync('bash', [path.join(__dirname, 'snapshot.sh')], { stdio: 'inherit' });

const logPath = path.join(process.env.RUNNER_TEMP || '.', 'runner_diag_sampler.log');
const out = fs.openSync(logPath, 'w');
const child = spawn('bash', [path.join(__dirname, 'sampler.sh')], {
  detached: true,
  stdio: ['ignore', out, out],
});
child.unref();

fs.appendFileSync(process.env.GITHUB_STATE, `samplerPid=${child.pid}\nsamplerLog=${logPath}\n`);
