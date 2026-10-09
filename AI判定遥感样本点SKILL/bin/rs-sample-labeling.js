#!/usr/bin/env node
// rs-sample-labeling.js — npx entry: `npx rs-sample-labeling install [--dest <dir>]`.
const cp = require('child_process');
const path = require('path');
const args = process.argv.slice(2);
const r = cp.spawnSync(process.execPath, [path.join(__dirname, '..', 'scripts', 'install.cjs'), ...args],
  { stdio: 'inherit' });
process.exit(r.status ?? 1);
