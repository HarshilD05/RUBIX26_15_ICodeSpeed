/**
 * Shared Memory Frame Reader
 * Reads camera frames from memory-mapped file created by Python AI process
 * Uses built-in Node.js fs module (no native dependencies)
 */

const fs = require('fs');
const path = require('path');

class SharedMemoryFrameReader {
  constructor(mmapFilePath) {
    this.mmapFilePath = mmapFilePath;
    this.fd = null;
    this.isOpen = false;
    
    // Preview control flag file (4-byte mmap)
    this.flagFilePath = mmapFilePath.replace('.mmap', '_flag.mmap');
    this.flagFd = null;
    this.flagBuffer = Buffer.allocUnsafe(4);
    
    // Frame header constants (must match Python SharedFrameBuffer)
    this.HEADER_SIZE = 24; // 6 * 4 bytes (6 uint32 values)
    this.MAX_FRAME_SIZE = 1920 * 1080 * 3; // HD frame
    this.TOTAL_SIZE = this.HEADER_SIZE + this.MAX_FRAME_SIZE;
    
    // Reusable buffer for reading
    this.headerBuffer = Buffer.allocUnsafe(this.HEADER_SIZE);
  }

  /**
   * Open the memory-mapped file
   * @returns {boolean} Success status
   */
  open() {
    try {
      // Check if file exists
      if (!fs.existsSync(this.mmapFilePath)) {
        console.error(`[SharedMemory] File not found: ${this.mmapFilePath}`);
        return false;
      }

      // Open file descriptor (read-only)
      this.fd = fs.openSync(this.mmapFilePath, 'r');
      
      // Get file stats
      const stats = fs.fstatSync(this.fd);
      if (stats.size < this.TOTAL_SIZE) {
        console.error(`[SharedMemory] File too small: ${stats.size} < ${this.TOTAL_SIZE}`);
        this.close();
        return false;
      }

      this.isOpen = true;
      console.log(`[SharedMemory] Opened: ${this.mmapFilePath}`);
      return true;

    } catch (error) {
      console.error(`[SharedMemory] Failed to open:`, error);
      this.close();
      return false;
    }
  }

  /**
   * Read frame header metadata
   * @returns {Object|null} Frame info or null if invalid
   */
  readFrameInfo() {
    if (!this.isOpen || this.fd === null) {
      return null;
    }

    try {
      // Read header from file
      fs.readSync(this.fd, this.headerBuffer, 0, this.HEADER_SIZE, 0);

      // Parse 6 uint32 values from header
      const width = this.headerBuffer.readUInt32LE(0);
      const height = this.headerBuffer.readUInt32LE(4);
      const channels = this.headerBuffer.readUInt32LE(8);
      const timestamp_sec = this.headerBuffer.readUInt32LE(12);
      const timestamp_usec = this.headerBuffer.readUInt32LE(16);
      const frame_size = this.headerBuffer.readUInt32LE(20);

      // Validate frame data
      if (width === 0 || height === 0 || frame_size === 0) {
        return null; // No frame written yet
      }

      if (frame_size > this.MAX_FRAME_SIZE) {
        console.error(`[SharedMemory] Invalid frame size: ${frame_size}`);
        return null;
      }

      const timestamp = timestamp_sec + (timestamp_usec / 1_000_000);

      return {
        width,
        height,
        channels,
        timestamp,
        frame_size
      };

    } catch (error) {
      console.error(`[SharedMemory] Error reading frame info:`, error);
      return null;
    }
  }

  /**
   * Read complete frame with pixel data
   * @returns {Object|null} Frame object with data and metadata
   */
  readFrame() {
    const info = this.readFrameInfo();
    if (!info) {
      return null;
    }

    try {
      // Read raw frame data (BGR format) from file
      const frameData = Buffer.allocUnsafe(info.frame_size);
      fs.readSync(this.fd, frameData, 0, info.frame_size, this.HEADER_SIZE);

      return {
        width: info.width,
        height: info.height,
        channels: info.channels,
        timestamp: info.timestamp,
        data: frameData // Raw BGR bytes
      };

    } catch (error) {
      console.error(`[SharedMemory] Error reading frame:`, error);
      return null;
    }
  }

  /**
   * Read frame as JPEG base64 for renderer
   * Python already writes JPEG, we just need to read and encode
   * @returns {Object|null} Frame with base64 JPEG data
   */
  readFrameAsJPEG() {
    const frame = this.readFrame();
    if (!frame) {
      return null;
    }

    // Frame data is already JPEG from Python, just base64 encode it
    return {
      width: frame.width,
      height: frame.height,
      timestamp: frame.timestamp,
      data: frame.data.toString('base64'),
      format: 'jpeg'
    };
  }

  /**
   * Open flag file for preview control
   * @returns {boolean} Success status
   */
  openFlagFile() {
    try {
      if (!fs.existsSync(this.flagFilePath)) {
        console.warn(`[SharedMemory] Flag file not found: ${this.flagFilePath}`);
        return false;
      }

      this.flagFd = fs.openSync(this.flagFilePath, 'r+');
      console.log(`[SharedMemory] Flag file opened: ${this.flagFilePath}`);
      return true;
    } catch (error) {
      console.error(`[SharedMemory] Failed to open flag file:`, error);
      return false;
    }
  }

  /**
   * Enable preview mode (Python will write frames to shared memory)
   * @returns {boolean} Success status
   */
  enablePreview() {
    try {
      if (this.flagFd === null) {
        if (!this.openFlagFile()) {
          return false;
        }
      }

      // Write 1 to flag file
      const buffer = Buffer.allocUnsafe(4);
      buffer.writeUInt32LE(1, 0);
      fs.writeSync(this.flagFd, buffer, 0, 4, 0);
      console.log('[SharedMemory] Preview enabled');
      return true;
    } catch (error) {
      console.error('[SharedMemory] Failed to enable preview:', error);
      return false;
    }
  }

  /**
   * Disable preview mode (Python will skip writing frames)
   * @returns {boolean} Success status
   */
  disablePreview() {
    try {
      if (this.flagFd === null) {
        if (!this.openFlagFile()) {
          return false;
        }
      }

      // Write 0 to flag file
      const buffer = Buffer.allocUnsafe(4);
      buffer.writeUInt32LE(0, 0);
      fs.writeSync(this.flagFd, buffer, 0, 4, 0);
      console.log('[SharedMemory] Preview disabled');
      return true;
    } catch (error) {
      console.error('[SharedMemory] Failed to disable preview:', error);
      return false;
    }
  }

  /**
   * Close the memory-mapped file
   */
  close() {
    try {
      if (this.fd !== null) {
        fs.closeSync(this.fd);
        this.fd = null;
      }
      if (this.flagFd !== null) {
        fs.closeSync(this.flagFd);
        this.flagFd = null;
      }
      this.isOpen = false;
      console.log('[SharedMemory] Closed');
    } catch (error) {
      console.error('[SharedMemory] Error closing:', error);
    }
  }
}

module.exports = SharedMemoryFrameReader;
