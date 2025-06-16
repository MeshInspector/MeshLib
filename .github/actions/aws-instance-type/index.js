// https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instancedata-data-retrieval.html

const core = require('@actions/core');
const http = require('http');

const EC2_INSTANCE_METADATA_HOST = '169.254.169.254';

function makeRequest(options) {
  return new Promise((resolve, reject) => {
    const req = http.request(options, (res) => {
      if (res.statusCode < 200 || res.statusCode >= 300) {
        return reject(new Error(`HTTP status code ${res.statusCode}`));
      }

      let data = '';
      res.on('data', (chunk) => {
        data += chunk;
      });
      res.on('end', () => {
        resolve(data);
      });
    });
    req.on('error', (err) => {
      reject(err);
    });
    req.on('timeout', () => {
      req.destroy();
      reject(new Error('Request timed out'));
    });
    req.end();
  });
}

async function run() {
  try {
    const token = await makeRequest({
      host: EC2_INSTANCE_METADATA_HOST,
      path: '/latest/api/token',
      method: 'PUT',
      headers: { 'X-aws-ec2-metadata-token-ttl-seconds': '60' },
      timeout: 2000,
    });

    const instanceType = await makeRequest({
      host: EC2_INSTANCE_METADATA_HOST,
      path: '/latest/meta-data/instance-type',
      method: 'GET',
      headers: { 'X-aws-ec2-metadata-token': token },
      timeout: 2000,
    });

    core.info(`${instanceType}`)
    core.setOutput('aws_instance_type', instanceType);
  } catch (error) {
    core.info('Could not retrieve AWS instance metadata');
    core.info('Falling back to "self-hosted"');
    core.setOutput('aws_instance_type', 'self-hosted');
  }
}

run().catch(error => {
    core.setFailed(`${error.message}`);
});
