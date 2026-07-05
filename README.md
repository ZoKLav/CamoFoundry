# Camo Foundry

Camo Foundry is a simple Windows desktop app for generating procedural camouflage patterns as high-resolution 2048×2048 images.

It is built for artists, modders, game devs, texture work, mockups, and general pattern generation. Pick a camo style, tune the sliders, adjust the palette, and export the result as a PNG or JPG.

## Features

### Procedural camo generation

Camo Foundry generates 2048×2048 camouflage textures directly from adjustable parameters. No external texture packs are required.

Supported pattern modes include:

* Blob / Organic
* Woodland
* Tiger Stripe
* Spray Paint
* Digital
* Flecktarn
* Splinter
* Topographic
* Rain Streak
* Brush Stroke

### Large control set

Most generation settings are exposed as sliders or toggles, so the app is easy to experiment with without editing config files.

Adjustable controls include:

* Scale
* Detail
* Density
* Contrast
* Roughness
* Edge softness
* Stripe width
* Stripe spacing
* Stripe wiggle
* Digital block size
* Dot size
* Speckle
* Grit / noise
* HSV jitter
* Color bleed
* Rotation

Additional toggles include:

* Tile-edge matching
* Invert color order
* Shuffle palette
* Outline passes
* Live preview
* Smooth preview scaling

### Color system

Camo Foundry includes common camouflage-style palettes and editable custom color slots.

Each custom color slot includes:

* Enable / disable toggle
* RGB sliders
* HSV sliders
* Color picker

This makes it easy to build anything from classic greens and browns to desert, snow, urban, sci-fi, high-contrast, or completely cursed experimental palettes.

### Export support

Generated patterns can be exported as:

* PNG
* JPG

Every export is generated at 2048×2048 resolution.

### Presets and settings

Settings can be saved and loaded as JSON files, making it easy to preserve a specific look, share generator settings, or return to a previous setup later.

## Fast start

To run from source without building a standalone executable:

```bat
run_dev.bat
```

This creates a local virtual environment, installs the required Python packages, and launches the app.

## Building the standalone Windows app

To build the standalone executable:

```bat
build.bat
```

The build script will:

1. Check that it is running on 64-bit Windows.
2. Look for an existing 64-bit Python installation.
3. Download and silently install 64-bit Python if needed.
4. Create a clean local virtual environment.
5. Install the required packages.
6. Build the app with PyInstaller.

After the build finishes, the executable will be located at:

```text
dist\CamoFoundry.exe
```

## Requirements

The build script handles the required Python setup automatically on Windows.

Main dependencies:

* Python 64-bit
* PySide6
* Pillow
* NumPy
* PyInstaller

## Notes

PyInstaller builds for the operating system it is running on. To produce the Windows `.exe`, run the build script on Windows.

Camo Foundry is intended for procedural art, texture generation, game assets, mockups, and visual design work. It is not tested or intended as field concealment equipment.

## License

Add your preferred license here.

Common choices:

* MIT License for permissive open-source use
* GPL if you want derivatives to stay open-source
* Private / All Rights Reserved if this is just for personal use

## Project status

Functional prototype.

The app currently supports multiple procedural camo styles, high-resolution export, live preview controls, custom palettes, and save/load settings.
