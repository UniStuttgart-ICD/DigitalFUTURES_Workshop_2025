#!/usr/bin/env python3
"""
Audio Test Script for Voice Demo Debugging
==========================================

This script tests the basic audio functionality to help debug why
you might not be hearing anything in the voice demo.

Usage:
    python test_audio.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os

def test_dependencies():
    """Test if required audio dependencies are available."""
    print("🔍 Testing audio dependencies...")
    
    try:
        import sounddevice as sd
        print("✅ sounddevice is available")
    except ImportError:
        print("❌ sounddevice is NOT available")
        print("   Install with: uv add sounddevice")
        return False
        
    try:
        import numpy as np
        print("✅ numpy is available")
    except ImportError:
        print("❌ numpy is NOT available")
        print("   Install with: uv add numpy")
        return False
        
    return True

def test_audio_devices():
    """Test available audio devices."""
    print("\n🔊 Testing audio devices...")
    
    try:
        import sounddevice as sd
        
        # List output devices
        devices = sd.query_devices()
        print(f"📱 Found {len(devices)} audio devices:")
        
        for i, device in enumerate(devices):
            if device['max_output_channels'] > 0:
                print(f"  {i}: {device['name']} (OUTPUT - {device['max_output_channels']} channels)")
        
        # Get default output device
        default_output = sd.default.device[1]
        print(f"\n🎯 Default output device: {default_output}")
        
        return True
        
    except Exception as e:
        print(f"❌ Audio device error: {e}")
        return False

def test_simple_beep():
    """Test playing a simple beep sound."""
    print("\n🎵 Testing simple audio playback...")
    
    try:
        import sounddevice as sd
        import numpy as np
        
        # Generate a simple 440Hz beep for 1 second
        duration = 1.0  # seconds
        sample_rate = 24000
        frequency = 440  # Hz
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        wave = 0.3 * np.sin(frequency * 2 * np.pi * t)
        
        print("🔊 Playing 1-second beep at 440Hz...")
        print("   (You should hear a tone now)")
        
        sd.play(wave, sample_rate)
        sd.wait()  # Wait for playback to complete
        
        print("✅ Beep playback completed")
        return True
        
    except Exception as e:
        print(f"❌ Beep test failed: {e}")
        return False

def test_audio_config():
    """Test the voice agent audio configuration."""
    print("\n⚙️ Testing voice agent audio config...")
    
    try:
        from ur.agents.voice_common.audio import AUDIO_CONFIG
        
        print("📋 Audio configuration:")
        for key, value in AUDIO_CONFIG.items():
            print(f"  {key}: {value}")
            
        # Test that all required keys exist
        required_keys = ['samplerate', 'channels', 'dtype']
        for key in required_keys:
            if key not in AUDIO_CONFIG:
                print(f"❌ Missing required key: {key}")
                return False
                
        print("✅ Audio configuration is valid")
        return True
        
    except Exception as e:
        print(f"❌ Audio config test failed: {e}")
        return False

def test_environment_variables():
    """Test relevant environment variables."""
    print("\n🌍 Testing environment variables...")
    
    # Key environment variables that affect audio
    env_vars = {
        'TEXT_ONLY_MODE': 'Should be false for audio',
        'OPENAI_API_KEY': 'Required for OpenAI voice',
        'DEBUG': 'Enable for detailed output',
    }
    
    for var, description in env_vars.items():
        value = os.getenv(var)
        if value:
            print(f"✅ {var}={value} ({description})")
        else:
            print(f"⚪ {var}=<not set> ({description})")
    
    # Check if TEXT_ONLY_MODE is explicitly disabling audio
    text_only = os.getenv('TEXT_ONLY_MODE', '').lower()
    if text_only in ['true', '1', 'yes', 'on']:
        print("⚠️ WARNING: TEXT_ONLY_MODE is enabled - this disables audio!")
        print("   Set TEXT_ONLY_MODE=false to enable audio")
        return False
        
    return True

def main():
    """Run all audio tests."""
    print("🔧 AUDIO DEBUGGING TEST SUITE")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Audio Devices", test_audio_devices),
        ("Audio Configuration", test_audio_config),
        ("Environment Variables", test_environment_variables),
        ("Simple Beep Test", test_simple_beep),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")
    
    print(f"\n{'=' * 50}")
    print(f"🏆 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Audio should work correctly.")
        print("\n💡 If you still can't hear audio in the demo, try:")
        print("   1. Check your speaker volume")
        print("   2. Ensure no other app is using your audio device")
        print("   3. Run the demo with DEBUG=true for more details")
    else:
        print("⚠️ Some tests failed. Fix the issues above before running the voice demo.")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 