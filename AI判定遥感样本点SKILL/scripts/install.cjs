#!/usr/bin/env node
// install.cjs — installer for rs-sample-labeling ("npx rs-sample-labeling install").
// Node >= 18, no dependencies. Copies SKILL.md + scripts/ + docs/(zh) into a skills dir.
const fs = require('fs');
const path = require('path');
const os = require('os');

const SKILL = 'rs-sample-labeling';
const FILES = ['SKILL.md', '.skill.json', 'README.md', 'README.zh-CN.md', 'package.json'];
const DIRS = ['scripts', 'docs', 'assets', 'bin'];

function destDefault() {
  const home = os.homedir();
  const cands = [
    path.join(home, '.zcode', 'skills', SKILL),
    path.join(home, '.codex', 'skills', SKILL),
  ];
  for (const c of cands) { if (fs.existsSync(path.dirname(c))) return c; }
  return cands[0];
}

function copyDir(src, dst) {
  fs.mkdirSync(dst, { recursive: true });
  for (const e of fs.readdirSync(src, { withFileTypes: true })) {
    if (e.name === '__pycache__' || e.name.endsWith('.pyc')) continue;
    const s = path.join(src, e.name), t = path.join(dst, e.name);
    if (e.isDirectory()) copyDir(s, t);
    else fs.copyFileSync(s, t);
  }
}

function install(target) {
  const root = path.resolve(__dirname, '..');
  fs.mkdirSync(target, { recursive: true });
  for (const f of FILES) {
    const s = path.join(root, f);
    if (fs.existsSync(s)) fs.copyFileSync(s, path.join(target, f));
  }
  for (const dd of DIRS) {
    const s = path.join(root, dd);
    if (fs.existsSync(s) && fs.statSync(s).isDirectory()) copyDir(s, path.join(target, dd));
  }
  // Back-compat: some consumers look for references/ — keep a copy if docs/ exists.
  const from = path.join(target, 'docs'), to = path.join(target, 'references');
  if (fs.existsSync(from) && !fs.existsSync(to)) copyDir(from, to);
  console.log(`installed ${SKILL} -> ${target}`);
  console.log('python deps: pillow numpy pandas | rasterio (s2_timeseries) | geopandas pyogrio fiona (points_to_shp)');
}

function usage() {
  console.log('usage: npx rs-sample-labeling install [--dest <dir>]');
}

function main() {
  const args = process.argv.slice(2);
  const cmd = args[0];
  if (!cmd || cmd === '-h' || cmd === '--help') { usage(); process.exit(cmd ? 0 : 1); }
  if (cmd !== 'install') { usage(); process.exit(1); }
  const i = args.indexOf('--dest');
  const target = i >= 0 && args[i + 1] ? path.resolve(args[i + 1]) : destDefault();
  install(target);
}
main();
