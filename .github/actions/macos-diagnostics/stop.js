const fs = require('fs');

const pid = Number(process.env.STATE_samplerPid);
if (pid) {
  // detached spawn made the sampler a group leader; kill the whole group
  try { process.kill(-pid); } catch { try { process.kill(pid); } catch {} }
}

const logPath = process.env.STATE_samplerLog;
if (logPath && fs.existsSync(logPath)) {
  console.log('::group::power/thermal/load samples');
  console.log(fs.readFileSync(logPath, 'utf8'));
  console.log('::endgroup::');
}
