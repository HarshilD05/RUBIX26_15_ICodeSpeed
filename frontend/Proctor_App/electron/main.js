const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');

// Determine if running in development or production
const isDev = process.env.NODE_ENV === 'development' || !app.isPackaged;

let mainWindow = null;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1200,
    minHeight: 700,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
      // Enable web security
      webSecurity: true,
      // Allow media access for webcam
      enableWebSQL: false,
    },
    titleBarStyle: 'default',
    backgroundColor: '#0f172a',
    show: false, // Don't show until ready
    icon: path.join(__dirname, '../public/vite.svg')
  });

  // Show window when ready to avoid visual flash
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  // Load the app
  if (isDev) {
    // In development, load from Vite dev server
    mainWindow.loadURL('http://localhost:5173');
    mainWindow.webContents.openDevTools();
  } else {
    // In production, load the built files
    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  // Handle any unhandled errors
  mainWindow.webContents.on('crashed', () => {
    console.error('Window crashed');
  });
}

// IPC handlers - add your custom handlers here
ipcMain.handle('getVersion', () => {
  return app.getVersion();
});

// App lifecycle
app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// Handle app quitting
app.on('before-quit', () => {
  // Cleanup if needed
});

// Security: Prevent navigation to external URLs
app.on('web-contents-created', (event, contents) => {
  contents.on('will-navigate', (event, navigationUrl) => {
    const parsedUrl = new URL(navigationUrl);
    
    // Allow navigation only to localhost in dev mode
    if (isDev && parsedUrl.origin === 'http://localhost:5173') {
      return;
    }
    
    // Prevent all other navigation
    if (parsedUrl.origin !== 'file://') {
      event.preventDefault();
    }
  });
});
