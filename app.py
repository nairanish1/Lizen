import streamlit as st 
import os 
import numpy as np
import tempfile
import soundfile as sf
from tempfile import NamedTemporaryFile
from Lizen import binaural_rendering, find_nearest
from scipy.signal import convolve
from scipy.fft import fft
import matplotlib.pyplot as plt
import librosa 
import librosa.display
import numpy as np
import plotly.graph_objects as go 
import io 
import wave
import sofa
from io import BytesIO
#--- Library Dependencies---#

st.set_page_config(
    page_title="🎧 Lizen: Binaural Audio Analyzer",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom color scheme
colors = {
    "background": "#F0F2F6",
    "text": "#1E1E1E",
    "primary": "#1E88E5",
    "secondary": "#43A047",
    "accent": "#FFC107"
}

# Custom CSS
st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: {colors['background']};
        color: {colors['text']};
    }}
    .sidebar .sidebar-content {{
        background: {colors['primary']};
    }}
    h1, h2, h3 {{
        color: {colors['primary']};
    }}
    .stButton>button {{
        color: {colors['background']};
        background-color: {colors['secondary']};
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }}
    .stButton>button:hover {{
        background-color: {colors['primary']};
    }}
    .stTextInput>div>div>input {{
        color: {colors['text']};
    }}
    .stSelectbox>div>div>select {{
        color: {colors['text']};
    }}
    .stSlider>div>div>div>div {{
        background-color: {colors['secondary']};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)
# App title and description
st.title("🎧 Lizen: Binaural Audio Analyzer")
st.markdown(
    """
    <div style='background-color: #E3F2FD; padding: 1rem; border-radius: 0.5rem;'>
    <h4 style='color: #1E88E5; margin-bottom: 0.5rem;'>Welcome to Lizen</h4>
    <p style='margin-bottom: 0; color: #000000; font-weight: bold;'>Explore the intricacies of binaural audio through advanced analytical visualizations. 
    Lizen provides a comprehensive suite of tools for rendering, analyzing, and understanding spatial audio.</p>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("<br><br>", unsafe_allow_html=True)



#system agnostic temporary directory
def save_uploaded_file(uploaded_file):
    try:
        #Use the system default temporary directory 
        temp_dir = tempfile.gettempdir()
        # Create a new temporary file in the specified directory
        with tempfile.NamedTemporaryFile(delete=False, suffix='-' + uploaded_file.name, dir=temp_dir) as tmp:
            # Write the contents of the uploaded file to the new file
            tmp.write(uploaded_file.read())
            # Return the path to the saved file
            return tmp.name
    except Exception as e:
        print(f"Failed to save file {uploaded_file.name}: {e}")
        return None

def get_sofa_positions(sofa_file_path):
    try:
        HRTF = sofa.Database.open(sofa_file_path) #opens the SOFA dataset 
        positions = HRTF.Source.Position.get_values(system="spherical") #
        return positions
    except Exception as e:
        st.error("Failed to load SOFA positions: " + str(e))
        return None
    
def load_source_files():
    st.sidebar.header("🛠️ Binaural Audio Configuration")
    source_file_paths = []
    sofa_choice = st.sidebar.radio("Choose HRTF Dataset:", ("Use Default SOFA Files", "Upload Your Own SOFA File"))
    if sofa_choice == "Use Default SOFA Files":
        default_sofa_files = os.listdir('/Users/anishnair/Project_Studio_Tech/Lizen_app/Sofa-Far-Field')
        selected_sofa_file = st.sidebar.selectbox("Select SOFA files", default_sofa_files)
        st.session_state.sofa_file_path = os.path.join('/Users/anishnair/Project_Studio_Tech/Lizen_app/Sofa-Far-Field', selected_sofa_file)
    else:
        uploaded_sofa_file = st.sidebar.file_uploader("Upload SOFA file", type=['sofa'])
        if uploaded_sofa_file:
            st.session_state.sofa_file_path = save_uploaded_file(uploaded_sofa_file)
    if not hasattr(st.session_state, "sofa_file_path"):
        st.error("Please select a SOFA file before proceeding")

    source_choice = st.sidebar.radio("Choose Audio Sources", ("Use Default Source Files", "Upload Your Own Audio Files"))
    if source_choice == "Use Default Source Files":
        default_source_files = os.listdir('/Users/anishnair/Project_Studio_Tech/Lizen_app/48k-Sounds')
        selected_source_files = st.sidebar.multiselect("Select Source Files", default_source_files)
        source_file_paths = [os.path.join('/Users/anishnair/Project_Studio_Tech/Lizen_app/48k-Sounds', file) for file in selected_source_files]
    elif source_choice == "Upload Your Own Audio Files":
        uploaded_source_files = st.sidebar.file_uploader("Upload Source Files", type=["wav", "mp3"], accept_multiple_files=True)
        if uploaded_source_files:
            source_file_paths = [save_uploaded_file(f) for f in uploaded_source_files]
    st.sidebar.markdown("Developed by Anish Nair 👷 🎧")
    return source_file_paths

def configure_azimuth_and_elevation(source_file_paths, positions):
    angle_positions = []
    elevation_positions = []
    if source_file_paths and positions is not None:
        for index, file_name in enumerate(source_file_paths):
            file_name = os.path.basename(file_name)
            st.subheader(f"Change Azimuth and Elevation Values for {file_name}")
            
            # Initialize session state for sliders if not already set
            if f"az_{index}" not in st.session_state:
                st.session_state[f"az_{index}"] = 0
            if f"el_{index}" not in st.session_state:
                st.session_state[f"el_{index}"] = 0

            # Use session state values as default for sliders
            user_azimuth = st.slider(
                f"**Azimuth of {file_name} in degrees**", 0, 360, 
                value=st.session_state[f"az_{index}"], key=f"az_{index}"
            )
            user_elevation = st.slider(
                f"**Elevation of {file_name} in degrees**", -90, 90, 
                value=st.session_state[f"el_{index}"], key=f"el_{index}"
            )

            # Find the nearest azimuth and elevation in the SOFA dataset
            azimuths = positions[:, 0]  # Assuming azimuths are the first column
            elevations = positions[:, 1]  # Assuming elevations are the second column

            nearest_azimuth, az_idx = find_nearest(azimuths, user_azimuth)
            nearest_elevation, el_idx = find_nearest(elevations, user_elevation)

            angle_positions.append(nearest_azimuth)
            elevation_positions.append(nearest_elevation)
            st.write(f"**Angle Approximation -**  Nearest azimuth: **{nearest_azimuth}**, elevation: **{nearest_elevation}**")

        return angle_positions, elevation_positions
    else:
        st.warning("Please select or upload at least one source file to proceed")
        return [], []

def int16_to_byte(audio_int16, sample_rate):
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(2)  # Stereo
        wf.setsampwidth(2)  # 2 bytes for int16
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    buffer.seek(0)
    return buffer.read()

def normalize_and_convert_to_int16(audio):
    # Ensure audio is float64 for precise calculations
    audio = audio.astype(np.float64)
    # Normalize the audio to [-1, 1]
    audio = audio / np.max(np.abs(audio))
    # Clip the values to ensure they do not exceed the range
    audio_clipped = np.clip(audio, -1, 1)
    # Convert to 16-bit integer
    audio_int16 = np.int16(audio_clipped * 32767)
    return audio_int16

impulse_responses = {
    "Bright": '/Users/anishnair/Project_Studio_Tech/Lizen_app/IR/Bright_IR.wav',
    "Medium": '/Users/anishnair/Project_Studio_Tech/Lizen_app/IR/Medium_IR.wav',
    "Dark": '/Users/anishnair/Project_Studio_Tech/Lizen_app/IR/Dark_IR.wav'
}
def apply_reverb(audio, ir_path, intensity):
    try:
        ir, _ = sf.read(ir_path)
        
        # Ensure IR is 2D (stereo) if audio is stereo
        if len(audio.shape) == 2 and len(ir.shape) == 1:
            ir = np.column_stack((ir, ir))
        
        # Perform convolution for each channel
        if len(audio.shape) == 2:
            wet_audio = np.zeros_like(audio)
            for i in range(audio.shape[1]):
                wet_audio[:, i] = convolve(audio[:, i], ir[:, i], mode="full")[:len(audio)]
        else:
            wet_audio = convolve(audio, ir, mode="full")[:len(audio)]
        
        output_audio = (1-intensity)*audio + intensity * wet_audio
        return output_audio
    except Exception as e:
        st.error(f"Error applying reverb: {e}")
        return audio

def melspectrogram(audio, sr):
    mel_spectrogram = librosa.feature.melspectrogram(y = audio, sr=sr, n_mels = 128, fmax=8000)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel Spectrogram For Processed Audio')
    return fig

def plot_interactive_spherical_coordinates(azimuths, elevations, distances):
    # Convert to radians
    azimuths_rad = np.radians(azimuths)
    elevations_rad = np.radians(elevations)

    # Convert spherical coordinates to Cartesian coordinates
    # This creates a single vector representing both azimuth and elevation
    x = distances * np.cos(elevations_rad) * np.sin(azimuths_rad)
    y = distances * np.cos(elevations_rad) * np.cos(azimuths_rad)
    z = distances * np.sin(elevations_rad)

    # Create a 3D scatter plot with Plotly
    fig = go.Figure()

    # Add combined azimuth and elevation vectors
    for i in range(len(x)):
        fig.add_trace(go.Scatter3d(
            x=[0, x[i]], y=[0, y[i]], z=[0, z[i]],
            mode='lines+markers',
            marker=dict(size=6, color='red'),
            line=dict(color='red', width=5),
            name=f'HRTF Vector {i+1}'
        ))

    # Create a sphere to represent the head
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))

    fig.add_trace(go.Surface(
        x=x_sphere, y=y_sphere, z=z_sphere,
        opacity=0.1, colorscale='gray', showscale=False,
        hoverinfo='skip'
    ))

    # Set layout for better visualization
    fig.update_layout(
        scene=dict(
            xaxis_title='X (Right-Left)',
            yaxis_title='Y (Front-Back)',
            zaxis_title='Z (Up-Down)',
            aspectmode='data'
        ),
        title="Interactive Source Localization Vector(s) Representation",
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Add annotations to explain the coordinate system
    fig.add_annotation(
        x=0.95, y=0.95, xref="paper", yref="paper",
        text="Red vectors show Source directions<br>combining azimuth and elevation",
        showarrow=False,
        font=dict(size=10),
        align="right",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1,
        borderpad=4
    )

    return fig
    
def get_hrtf(sofa_file, angle, elevation, target_fs=48000):
    HRTF = sofa.Database.open(sofa_file)
    fs_H = HRTF.Data.SamplingRate.get_values()[0]
    positions = HRTF.Source.Position.get_values(system="spherical")
    
    # Adjust angle to match database convention 
    angle = 360 - angle
    if angle == 360:
        angle = 0
    
    # Retrieve HRTF data for the angle and the elevation 
    [az, _] = find_nearest(positions[:,0], angle)
    az_indices = np.where(positions[:,0] == az)[0]
    [el, el_idx] = find_nearest(positions[az_indices][:,1], elevation)
    M_idx = az_indices[el_idx]

    H = np.zeros([HRTF.Dimensions.N, 2])
    H[:, 0] = HRTF.Data.IR.get_values(indices={"M": M_idx, "R":0, "E":0})
    H[:, 1] = HRTF.Data.IR.get_values(indices={"M": M_idx, "R":1, "E":0})

    if fs_H != target_fs:
        H = librosa.core.resample(H.transpose(), fs_H, target_fs, res_type="kaiser_best").transpose()
    
    return H, target_fs


def plot_hrtf(sofa_file, angle, elevation, file_name, target_fs=48000):
    H, fs = get_hrtf(sofa_file, angle, elevation, target_fs)
    
    n = H.shape[0]
    freq = np.fft.rfftfreq(n, d=1/fs)
    hrtf_left = 20 * np.log10(np.abs(np.fft.rfft(H[:, 0])))
    hrtf_right = 20 * np.log10(np.abs(np.fft.rfft(H[:, 1])))
    
    plt.figure(figsize=(12, 6))
    plt.semilogx(freq, hrtf_left, label='Left Ear')
    plt.semilogx(freq, hrtf_right, label='Right Ear')
    plt.title(f'{file_name} HRTF for Azimuth: {angle}°, Elevation: {elevation}°')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend()
    plt.grid(True)
    plt.xlim(20, fs/2)
    plt.tight_layout()
    return plt.gcf()


def plot_itd(sofa_file, angles, elevation, target_fs=48000):
    itds = []
    for angle in angles:
        H, fs = get_hrtf(sofa_file, angle, elevation, target_fs)
        correlation = np.correlate(H[:, 0], H[:, 1], mode='full')
        delay = np.argmax(correlation) - (len(H) - 1)
        itd = delay / fs * 1000  # Convert to milliseconds
        itds.append(itd)
    
    plt.figure(figsize=(10, 5))
    plt.plot(angles, itds)
    plt.title(f'Interaural Time Difference (ITD) for Elevation: {elevation}°')
    plt.xlabel('Azimuth (degrees)')
    plt.ylabel('ITD (ms)')
    plt.grid(True)
    plt.tight_layout()
    return plt.gcf()

def plot_ild(sofa_file, angles, elevation, target_fs=48000):
    ilds = []
    for angle in angles:
        H, fs = get_hrtf(sofa_file, angle, elevation, target_fs)
        left_power = np.sum(H[:, 0]**2)
        right_power = np.sum(H[:, 1]**2)
        ild = 10 * np.log10(right_power / left_power)
        ilds.append(ild)
    
    plt.figure(figsize=(10, 5))
    plt.plot(angles, ilds)
    plt.title(f'Interaural Level Difference (ILD) for Elevation: {elevation}°')
    plt.xlabel('Azimuth (degrees)')
    plt.ylabel('ILD (dB)')
    plt.grid(True)
    plt.tight_layout()
    return plt.gcf()

#LOAD
source_file_paths = load_source_files()
source_file_name = os.path.basename(st.session_state.sofa_file_path)
positions = get_sofa_positions(st.session_state.sofa_file_path if hasattr(st.session_state, "sofa_file_path") else None)
angle_positions, elevation_positions = configure_azimuth_and_elevation(source_file_paths, positions)

if source_file_paths and hasattr(st.session_state, "sofa_file_path"): 
    selected_reverb = st.selectbox("Select your Reverb Type", list(impulse_responses.keys()))
    reverb_intensity = st.slider("Reverb Intensity (Dry to Wet)", 0.0, 1.0, 0.0)

    if st.button("🚀 Render Binaural Audio"):
        st.success("Audio rendered successfully!")
        st.subheader("Binaural Rendered Audio:")
        stereo_audio, fs = binaural_rendering(st.session_state.sofa_file_path, source_file_paths, angle_positions, elevation_positions)
        if selected_reverb in impulse_responses:
            stereo_audio = apply_reverb(stereo_audio, impulse_responses[selected_reverb], reverb_intensity)
        stereo_audio_int16 = normalize_and_convert_to_int16(stereo_audio)
        audio_bytes = int16_to_byte(stereo_audio_int16, fs)
        st.audio(audio_bytes, format="audio/wav")

        #Create name for downloading audio
        if len(source_file_paths) == 1:
            original_file_name = os.path.basename(source_file_paths[0])
            base_name, _ = os.path.splitext(original_file_name)
            new_file_name = f"{base_name}_binaural_audio.wav"
        else:
            base_names = [os.path.splitext(os.path.basename(path))[0] for path in source_file_paths]
            new_file_name = f"multiple_sources_binaural_audio.wav"


        #Adding Download Binaural Button 
        st.download_button(
        label="Download Binaural Audio",
        data=audio_bytes,
        file_name=new_file_name,
        mime="audio/wav"
    )
        def fig_to_bytes(fig):
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            return buf.getvalue()
        
         #plot spherical coordinates 
        distances = [1] * len(angle_positions) #assuming unit distances for all sources
        fig = plot_interactive_spherical_coordinates(angle_positions, elevation_positions, distances)
        st.plotly_chart(fig)

        st.header("HRTF Analysis:")
        for i, (azimuth, elevation) in enumerate(zip(angle_positions, elevation_positions)):
            hrtf_fig = plot_hrtf(st.session_state.sofa_file_path, azimuth, elevation, source_file_name)
            if hrtf_fig:
                st.pyplot(hrtf_fig)
                hrtf_bytes = fig_to_bytes(hrtf_fig)
                st.download_button(
                label=f"Download HRTF Analysis (Azimuth: {azimuth}, Elevation: {elevation})",
                data=hrtf_bytes,
                file_name=f"hrtf_analysis_az{azimuth}_el{elevation}.png",
                mime="image/png",
                key=f"download_hrtf_{i}"
            )

        st.header("Interaural Time Difference (ITD):")
        angles = np.arange(0, 361, 5)  # Angles from 0 to 360 in steps of 5 degrees
        itd_fig = plot_itd(st.session_state.sofa_file_path, angles, elevation_positions[0])
        st.pyplot(itd_fig)
        itd_bytes = fig_to_bytes(itd_fig)
        st.download_button(
        label="Download ITD Analysis",
        data=itd_bytes,
        file_name="itd_analysis.png",
        mime="image/png",
        key="download_itd"
    )
        st.header("Interaural Level Difference (ILD):")
        ild_fig = plot_ild(st.session_state.sofa_file_path, angles, elevation_positions[0])
        st.pyplot(ild_fig)
        ild_bytes = fig_to_bytes(ild_fig)
        st.download_button(
        label="Download ILD Analysis",
        data=ild_bytes,
        file_name="ild_analysis.png",
        mime="image/png",
        key="download_ild"
    )

        #analyze mel-spec
        st.header("Frequency Domain Analysis:")
        mono_audio = np.mean(stereo_audio, axis=1)
        mel_fig = melspectrogram(mono_audio, fs)
        st.pyplot(mel_fig)
        mel_bytes = fig_to_bytes(mel_fig)
        st.download_button(
        label="Download Mel-spectrogram",
        data=mel_bytes,
        file_name="mel_spectrogram.png",
        mime="image/png",
        key="download_mel"
    )
else: 
    st.warning("Please configure both SOFA and source files to proceed with audio rendering")








