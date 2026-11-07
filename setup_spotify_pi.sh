#!/bin/bash
# One-time Spotify setup on the Pi (no sudo), as in vcm-me2 docs/runtime.md.
#   bash ~/me3/setup_spotify_pi.sh install   # librespot -> ~/.local/bin
#   bash ~/me3/setup_spotify_pi.sh pair      # prints a spotify.com/pair link: approve it
# Pairing caches the login in ~/.cache/librespot/credentials.json; after that
# run_spotify.sh keeps the "Watson" device listed in the Web API.
set -e
case "$1" in
  install)
    url=$(curl -s https://api.github.com/repos/dtcooper/raspotify/releases/latest \
          | grep -o 'https://[^"]*/raspotify_[^"]*arm64\.deb' | head -1)
    [ -n "$url" ] || { echo "no arm64 raspotify release found"; exit 1; }
    tmp=$(mktemp -d)
    curl -sL -o "$tmp/raspotify.deb" "$url"
    dpkg-deb -x "$tmp/raspotify.deb" "$tmp/rs"
    mkdir -p ~/.local/bin ~/.cache/librespot
    cp "$tmp/rs/usr/bin/librespot" ~/.local/bin/librespot
    rm -rf "$tmp"
    ~/.local/bin/librespot --version
    ;;
  pair)
    # device pairing: no browser and no zeroconf needed on the Pi
    exec ~/.local/bin/librespot --name Watson --backend pulseaudio \
      --cache ~/.cache/librespot --enable-device-auth
    ;;
  *) echo "usage: $0 install|pair"; exit 2 ;;
esac
