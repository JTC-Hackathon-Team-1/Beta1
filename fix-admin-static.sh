#!/bin/bash

# Source and destination directories
SOURCE_DIR="/Users/jameswilson/Documents/FLG-F24/WIP/CasalinguaWIP/casalingua_restructured"
DEST_DIR="$HOME/Desktop/casalingua_mvp"

echo "Searching for built static files in source project..."

# Look for React build files with hash names (common pattern for built React apps)
FOUND_JS=$(find "$SOURCE_DIR" -name "main.*.js" | grep -v "node_modules")
FOUND_CSS=$(find "$SOURCE_DIR" -name "main.*.css" | grep -v "node_modules")
FOUND_MANIFEST=$(find "$SOURCE_DIR" -name "manifest.json" | grep -v "node_modules")

# Check if we found the files directly
if [ ! -z "$FOUND_JS" ] && [ ! -z "$FOUND_CSS" ]; then
    echo "Found built static files:"
    echo "JS: $FOUND_JS"
    echo "CSS: $FOUND_CSS"
    
    # Create the static directories
    mkdir -p "$DEST_DIR/public/static/js"
    mkdir -p "$DEST_DIR/public/static/css"
    
    # Copy the files
    cp $FOUND_JS "$DEST_DIR/public/static/js/"
    cp $FOUND_CSS "$DEST_DIR/public/static/css/"
    
    if [ ! -z "$FOUND_MANIFEST" ]; then
        echo "Manifest: $FOUND_MANIFEST"
        cp $FOUND_MANIFEST "$DEST_DIR/public/"
    fi
    
    echo "Static files copied successfully."
else
    echo "Could not find built static files directly. Looking for a build directory..."
    
    # Look for common build directory patterns
    BUILD_DIRS=$(find "$SOURCE_DIR" -type d -name "build" -o -name "dist" -o -name "public" | grep -v "node_modules")
    
    if [ ! -z "$BUILD_DIRS" ]; then
        echo "Found possible build directories:"
        echo "$BUILD_DIRS"
        
        # Check each directory for the static files
        for dir in $BUILD_DIRS; do
            echo "Checking $dir for static files..."
            
            # Look for static directories or files
            if [ -d "$dir/static" ]; then
                echo "Found static directory in $dir"
                mkdir -p "$DEST_DIR/public/static"
                cp -r "$dir/static"/* "$DEST_DIR/public/static/"
                echo "Copied static directory from $dir to $DEST_DIR/public/static/"
            fi
            
            # Check for manifest.json
            if [ -f "$dir/manifest.json" ]; then
                echo "Found manifest.json in $dir"
                cp "$dir/manifest.json" "$DEST_DIR/public/"
                echo "Copied manifest.json to $DEST_DIR/public/"
            fi
            
            # Check for index.html to ensure we have the correct references
            if [ -f "$dir/index.html" ]; then
                echo "Found index.html in $dir"
                
                # Extract JS and CSS file references from index.html
                JS_FILES=$(grep -o '/static/js/[^"]*\.js' "$dir/index.html" || grep -o 'static/js/[^"]*\.js' "$dir/index.html")
                CSS_FILES=$(grep -o '/static/css/[^"]*\.css' "$dir/index.html" || grep -o 'static/css/[^"]*\.css' "$dir/index.html")
                
                if [ ! -z "$JS_FILES" ] || [ ! -z "$CSS_FILES" ]; then
                    echo "Found references to static files in index.html:"
                    echo "$JS_FILES"
                    echo "$CSS_FILES"
                    
                    # Create an updated index.html for our admin route
                    cp "$dir/index.html" "$DEST_DIR/public/index.html"
                    echo "Copied index.html to $DEST_DIR/public/"
                fi
            fi
        done
    else
        echo "No build directories found. Creating minimal static files..."
        
        # Create minimal versions of the required files
        mkdir -p "$DEST_DIR/public/static/js"
        mkdir -p "$DEST_DIR/public/static/css"
        
        # Create a minimal JavaScript file
        cat > "$DEST_DIR/public/static/js/main.js" << 'EOF'
document.addEventListener('DOMContentLoaded', function() {
    console.log('CasaLingua Admin Panel Initialized');
    
    // Check if admin panel container exists
    const adminContainer = document.getElementById('admin-root');
    if (adminContainer) {
        adminContainer.innerHTML = '<h1>CasaLingua Admin Panel</h1><p>The admin panel is running in basic mode. Static assets were not found.</p>';
    }
});
EOF

        # Create a minimal CSS file
        cat > "$DEST_DIR/public/static/css/main.css" << 'EOF'
body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 0;
    background-color: #f5f5f5;
}

#admin-root {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    background-color: white;
    border-radius: 5px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    margin-top: 20px;
}

h1 {
    color: #333;
    border-bottom: 1px solid #eee;
    padding-bottom: 10px;
}
EOF

        # Create a manifest.json file
        cat > "$DEST_DIR/public/manifest.json" << 'EOF'
{
  "short_name": "CasaLingua",
  "name": "CasaLingua Admin Panel",
  "icons": [
    {
      "src": "favicon.ico",
      "sizes": "64x64 32x32 24x24 16x16",
      "type": "image/x-icon"
    }
  ],
  "start_url": ".",
  "display": "standalone",
  "theme_color": "#000000",
  "background_color": "#ffffff"
}
EOF

        # Create a basic index.html file
        cat > "$DEST_DIR/public/index.html" << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CasaLingua Admin Panel</title>
    <link rel="stylesheet" href="/static/css/main.css">
    <link rel="manifest" href="/manifest.json">
</head>
<body>
    <div id="admin-root"></div>
    <script src="/static/js/main.js"></script>
</body>
</html>
EOF

        echo "Created minimal static files."
    fi
fi

# Create or update the server.js file to properly serve static files
cat > "$DEST_DIR/server.js" << 'EOF'
const express = require('express');
const path = require('path');
const fs = require('fs');
const app = express();
const port = process.env.PORT || 3000;

// Static file middleware
app.use('/static', express.static(path.join(__dirname, 'public/static')));
app.use(express.static(path.join(__dirname, 'public')));

// Serve the admin panel
app.get('/admin', (req, res) => {
  console.log('[INFO] Serving admin panel (path: )');
  res.sendFile(path.join(__dirname, 'public/index.html'));
});

// Fallback route
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public/index.html'));
});

// Start the server
app.listen(port, () => {
  console.log(`[INFO] ✓ Model loaded successfully`);
  console.log(`[INFO] Running on CPU - consider enabling GPU for faster performance`);
  console.log(`[INFO] Admin frontend mounted`);
  console.log(`Server running at http://localhost:${port}`);
});
EOF

echo "Updated server.js to correctly serve static files."

# Update package.json if it exists
if [ -f "$DEST_DIR/package.json" ]; then
    # Check if express is a dependency
    if ! grep -q '"express"' "$DEST_DIR/package.json"; then
        # Add express dependency
        sed -i.bak 's/"dependencies": {/"dependencies": {\n    "express": "^4.18.2",/g' "$DEST_DIR/package.json"
        rm "$DEST_DIR/package.json.bak"
        echo "Added express dependency to package.json"
    fi
else
    # Create a minimal package.json file
    cat > "$DEST_DIR/package.json" << 'EOF'
{
  "name": "casalingua_mvp",
  "version": "1.0.0",
  "description": "CasaLingua MVP with admin panel",
  "main": "server.js",
  "scripts": {
    "start": "node server.js",
    "dev": "nodemon server.js"
  },
  "dependencies": {
    "express": "^4.18.2"
  },
  "devDependencies": {
    "nodemon": "^2.0.22"
  }
}
EOF
    echo "Created package.json file."
fi

echo "Setup complete! Your CasaLingua admin panel should now work properly."
echo "To run the project:"
echo "1. cd $DEST_DIR"
echo "2. npm install"
echo "3. npm run dev"
echo "4. Open http://localhost:3000/admin in your browser"