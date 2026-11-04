/// <reference lib="webworker" />
import { sha256 } from '@noble/hashes/sha2.js';
import { bytesToHex } from '@noble/hashes/utils.js';

self.onmessage = async (event: MessageEvent<{ file: File; chunkSize: number }>) => {
  try {
    const { file, chunkSize } = event.data;
    const whole = sha256.create(), chunks: string[] = [];
    for (let offset = 0; offset < file.size; offset += chunkSize) {
      const data = new Uint8Array(await file.slice(offset, offset + chunkSize).arrayBuffer());
      whole.update(data); chunks.push(bytesToHex(sha256(data)));
      self.postMessage({ progress: Math.min(1, (offset + data.length) / file.size) });
    }
    self.postMessage({ result: { sha256: bytesToHex(whole.digest()), chunks } });
  } catch (error) { self.postMessage({ error: error instanceof Error ? error.message : String(error) }); }
};
