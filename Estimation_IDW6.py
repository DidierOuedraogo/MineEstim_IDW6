import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import zipfile
import math
import base64
from stqdm import stqdm
import seaborn as sns
from PIL import Image
from io import BytesIO
import ezdxf
import trimesh
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import datetime
import tempfile
import os
import traceback

# Configuration de la page
st.set_page_config(
    page_title="MineEstim - Inverse Distance",
    page_icon="⛏️",
    layout="wide"
)

# Fonction pour capturer et afficher les erreurs de manière détaillée
def show_detailed_error(error_title, exception):
    st.error(error_title)
    st.write("**Détails de l'erreur:**")
    st.code(traceback.format_exc())
    st.write("**Type d'erreur:** ", type(exception).__name__)
    st.write("**Message d'erreur:** ", str(exception))

# Logo simple pour l'industrie minière
def create_mining_logo():
    # Créer une image avec un logo d'exploration minière simple
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    # Dessiner un casque de mineur
    circle = plt.Circle((5, 5), 2, fill=True, color='gold')
    ax.add_patch(circle)
    
    # Dessiner une pioche
    ax.plot([2, 4], [3, 7], 'k-', linewidth=3)
    ax.plot([3, 6], [8, 6], 'k-', linewidth=3)
    
    # Supprimer les axes
    ax.axis('off')
    
    # Convertir en image
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    buf.seek(0)
    return buf

# Générer des données synthétiques pour un gisement de cuivre
def generate_synthetic_copper_data(n_samples=300, random_seed=42):
    np.random.seed(random_seed)
    
    # Coordonnées centrales du gisement
    center_x, center_y, center_z = 5000, 5000, 100
    
    # Paramètres de taille du gisement
    x_range, y_range, z_range = 200, 100, 50
    
    # Générer les coordonnées aléatoires autour du centre avec distribution normale
    x = np.random.normal(center_x, x_range/3, n_samples)
    y = np.random.normal(center_y, y_range/3, n_samples)
    z = np.random.normal(center_z, z_range/3, n_samples)
    
    # Créer une distribution pour les teneurs de cuivre qui diminue avec la distance du centre
    distances = np.sqrt((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2)
    max_distance = np.max(distances)
    
    # Distribution de base de la teneur en cuivre (%)
    base_copper = 3.0 * (1 - distances / max_distance) + np.random.normal(0, 0.3, n_samples)
    copper_grades = np.clip(base_copper, 0.1, 5.0)  # Limiter les teneurs entre 0.1% et 5.0%
    
    # Créer un domaine géologique principal et des sous-domaines
    main_domain = np.ones(n_samples, dtype=int)
    sub_domains = np.random.choice(['A', 'B', 'C'], size=n_samples, p=[0.6, 0.3, 0.1])
    
    # Ajouter quelques échantillons de "fond" avec des teneurs plus faibles
    background_samples = int(n_samples * 0.2)
    bg_x = np.random.uniform(center_x - x_range*2, center_x + x_range*2, background_samples)
    bg_y = np.random.uniform(center_y - y_range*2, center_y + y_range*2, background_samples)
    bg_z = np.random.uniform(center_z - z_range*2, center_z + z_range*2, background_samples)
    bg_copper = np.random.uniform(0.01, 0.5, background_samples)
    bg_domain = np.zeros(background_samples, dtype=int)
    bg_sub_domains = np.random.choice(['D', 'E'], size=background_samples, p=[0.7, 0.3])
    
    # Combiner les échantillons principaux et de fond
    x = np.concatenate([x, bg_x])
    y = np.concatenate([y, bg_y])
    z = np.concatenate([z, bg_z])
    copper_grades = np.concatenate([copper_grades, bg_copper])
    domain = np.concatenate([main_domain, bg_domain])
    sub_domain = np.concatenate([sub_domains, bg_sub_domains])
    
    # Ajouter une colonne de densité - la densité varie avec la teneur de cuivre
    # Supposons une relation linéaire simple: densité = 2.5 + 0.2 * Cu%
    density = 2.5 + 0.2 * copper_grades + np.random.normal(0, 0.05, len(copper_grades))
    
    # Créer le DataFrame
    df = pd.DataFrame({
        'X': x,
        'Y': y,
        'Z': z,
        'CU_PCT': copper_grades,
        'DENSITY': density,
        'DOMAIN': domain,
        'SUB_DOMAIN': sub_domain
    })
    
    return df

# Fonction calculate_stats modifiée pour renvoyer des valeurs par défaut
def calculate_stats(values):
    if not values or len(values) == 0:
        return {
            'count': 0,
            'min': 0,
            'max': 0,
            'mean': 0,
            'median': 0,
            'stddev': 0,
            'variance': 0,
            'cv': 0
        }
    
    values = np.array(values)
    return {
        'count': len(values),
        'min': np.min(values),
        'max': np.max(values),
        'mean': np.mean(values),
        'median': np.median(values),
        'stddev': np.std(values),
        'variance': np.var(values),
        'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
    }

# Fonction pour générer un rapport PDF d'estimation
def generate_estimation_report(estimated_blocks, composites_data, idw_params, search_params, block_sizes, 
                              tonnage_data=None, plot_info=None, density_method="constant", density_value=2.7, 
                              density_column=None, project_name="Projet Minier"):
    try:
        # Créer un buffer pour le PDF
        buffer = BytesIO()
        
        # Créer le document PDF
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = styles['Title']
        heading1_style = styles['Heading1']
        heading2_style = styles['Heading2']
        normal_style = styles['Normal']
        
        # Liste des éléments à mettre dans le rapport
        elements = []
        
        # Titre et date
        elements.append(Paragraph(f"Rapport d'Estimation - {project_name}", title_style))
        elements.append(Paragraph(f"Date: {datetime.datetime.now().strftime('%d/%m/%Y')}", normal_style))
        elements.append(Paragraph(f"Auteur: Didier Ouedraogo, P.Geo", normal_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Méthodologie
        elements.append(Paragraph("1. Méthodologie d'Estimation", heading1_style))
        elements.append(Paragraph("Cette estimation a été réalisée par la méthode de l'inverse des distances "
                                 "qui attribue plus de poids aux échantillons proches qu'aux échantillons éloignés.", normal_style))
        elements.append(Spacer(1, 0.1*inch))
        
        # Paramètres
        elements.append(Paragraph("2. Paramètres d'Estimation", heading1_style))
        
        # Tableau des paramètres
        data = [
            ["Paramètre", "Valeur"],
            ["Méthode d'estimation", "Inverse des distances"],
            ["Puissance (p)", str(idw_params['power'])],
            ["Anisotropie X", str(idw_params['anisotropy']['x'])],
            ["Anisotropie Y", str(idw_params['anisotropy']['y'])],
            ["Anisotropie Z", str(idw_params['anisotropy']['z'])],
            ["Rayon de recherche X", str(search_params['x']) + " m"],
            ["Rayon de recherche Y", str(search_params['y']) + " m"],
            ["Rayon de recherche Z", str(search_params['z']) + " m"],
            ["Min. échantillons", str(search_params['min_samples'])],
            ["Max. échantillons", str(search_params['max_samples'])],
            ["Taille des blocs", f"{block_sizes['x']} × {block_sizes['y']} × {block_sizes['z']} m"]
        ]
        
        # Ajouter les informations sur la densité
        if density_method == "constant":
            data.append(["Densité", f"{density_value} t/m³ (constante)"])
        else:
            data.append(["Densité", f"Variable (colonne {density_column})"])
        
        # Créer le tableau
        t = Table(data, colWidths=[2.5*inch, 2*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(t)
        elements.append(Spacer(1, 0.2*inch))
        
        # Statistiques
        elements.append(Paragraph("3. Statistiques", heading1_style))
        
        # Statistiques des composites
        elements.append(Paragraph("3.1 Statistiques des Composites", heading2_style))
        composite_values = [comp['VALUE'] for comp in composites_data if 'VALUE' in comp]
        composite_stats = calculate_stats(composite_values)
        
        comp_data = [
            ["Statistique", "Valeur"],
            ["Nombre d'échantillons", str(composite_stats['count'])],
            ["Minimum", f"{composite_stats['min']:.3f}"],
            ["Maximum", f"{composite_stats['max']:.3f}"],
            ["Moyenne", f"{composite_stats['mean']:.3f}"],
            ["Médiane", f"{composite_stats['median']:.3f}"],
            ["Écart-type", f"{composite_stats['stddev']:.3f}"],
            ["CV", f"{composite_stats['cv']:.3f}"]
        ]
        
        comp_table = Table(comp_data, colWidths=[2.5*inch, 2*inch])
        comp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(comp_table)
        
        # Histogramme des composites si assez de données
        if composite_stats['count'] > 1:
            fig_comp, ax_comp = plt.subplots(figsize=(6, 4))
            n_bins = max(5, int(1 + 3.322 * math.log10(len(composite_values))))
            sns.histplot(composite_values, bins=n_bins, kde=True, color="darkblue", ax=ax_comp)
            ax_comp.set_title("Distribution des teneurs des composites")
            ax_comp.set_xlabel("Teneur")
            ax_comp.set_ylabel("Fréquence")
            
            # Sauvegarder l'histogramme dans un fichier temporaire
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                fig_comp.savefig(tmp_file.name, format='png', dpi=150, bbox_inches='tight')
                comp_hist_path = tmp_file.name
            
            # Ajouter l'histogramme au rapport
            elements.append(Spacer(1, 0.1*inch))
            comp_img = ReportLabImage(comp_hist_path, width=4*inch, height=3*inch)
            elements.append(comp_img)
            elements.append(Spacer(1, 0.2*inch))
        
        # Statistiques des blocs
        if estimated_blocks and len(estimated_blocks) > 0:
            elements.append(Paragraph("3.2 Statistiques du Modèle de Blocs", heading2_style))
            block_values = [block.get('value', 0) for block in estimated_blocks]
            block_stats = calculate_stats(block_values)
            
            block_data = [
                ["Statistique", "Valeur"],
                ["Nombre de blocs", str(block_stats['count'])],
                ["Minimum", f"{block_stats['min']:.3f}"],
                ["Maximum", f"{block_stats['max']:.3f}"],
                ["Moyenne", f"{block_stats['mean']:.3f}"],
                ["Médiane", f"{block_stats['median']:.3f}"],
                ["Écart-type", f"{block_stats['stddev']:.3f}"],
                ["CV", f"{block_stats['cv']:.3f}"]
            ]
            
            block_table = Table(block_data, colWidths=[2.5*inch, 2*inch])
            block_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(block_table)
            
            # Histogramme des blocs si assez de données
            if block_stats['count'] > 1:
                fig_block, ax_block = plt.subplots(figsize=(6, 4))
                n_bins = max(5, int(1 + 3.322 * math.log10(len(block_values))))
                sns.histplot(block_values, bins=n_bins, kde=True, color="teal", ax=ax_block)
                ax_block.set_title("Distribution des teneurs du modèle de blocs")
                ax_block.set_xlabel("Teneur")
                ax_block.set_ylabel("Fréquence")
                
                # Sauvegarder l'histogramme dans un fichier temporaire
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                    fig_block.savefig(tmp_file.name, format='png', dpi=150, bbox_inches='tight')
                    block_hist_path = tmp_file.name
                
                # Ajouter l'histogramme au rapport
                elements.append(Spacer(1, 0.1*inch))
                block_img = ReportLabImage(block_hist_path, width=4*inch, height=3*inch)
                elements.append(block_img)
                elements.append(Spacer(1, 0.2*inch))
            
            # Résumé global
            elements.append(Paragraph("3.3 Résumé Global", heading2_style))
            
            # Calculer les métriques globales
            if density_method == "constant":
                avg_density = density_value
            else:
                # Calculer la densité moyenne pondérée par le volume
                block_volumes = [block.get('size_x', 0) * block.get('size_y', 0) * block.get('size_z', 0) for block in estimated_blocks]
                avg_density = sum(block.get('density', density_value) * vol for block, vol in zip(estimated_blocks, block_volumes)) / (sum(block_volumes) or 1)
            
            if 'size_x' in estimated_blocks[0] and 'size_y' in estimated_blocks[0] and 'size_z' in estimated_blocks[0]:
                block_volume = estimated_blocks[0]['size_x'] * estimated_blocks[0]['size_y'] * estimated_blocks[0]['size_z']
                total_volume = len(estimated_blocks) * block_volume
                total_tonnage = total_volume * avg_density
            else:
                block_volume = 0
                total_volume = 0
                total_tonnage = 0
            
            summary_data = [
                ["Paramètre", "Valeur"],
                ["Nombre de blocs", f"{len(estimated_blocks):,}"],
                ["Teneur moyenne", f"{block_stats['mean']:.3f}"],
                ["Densité moyenne", f"{avg_density:.2f} t/m³"],
                ["Volume total (m³)", f"{total_volume:,.0f}"],
                ["Tonnage total (t)", f"{total_tonnage:,.0f}"]
            ]
            
            summary_table = Table(summary_data, colWidths=[2.5*inch, 2*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(summary_table)
            elements.append(Spacer(1, 0.2*inch))
        
        # Analyse Tonnage-Teneur
        if tonnage_data and plot_info and 'cutoffs' in tonnage_data and 'tonnages' in tonnage_data and 'grades' in tonnage_data:
            elements.append(Paragraph("4. Analyse Tonnage-Teneur", heading1_style))
            
            # Graphique Tonnage-Teneur
            if plot_info.get('method') != 'between':
                # Créer le graphique Tonnage-Teneur
                fig_tonnage, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
                
                # Graphique du tonnage
                ax1.plot(tonnage_data['cutoffs'], tonnage_data['tonnages'], 'b-', linewidth=2)
                ax1.set_xlabel('Teneur de coupure')
                ax1.set_ylabel('Tonnage (t)')
                ax1.set_title('Courbe Tonnage-Teneur')
                ax1.grid(True)
                
                # Graphique des teneurs moyennes
                ax2.plot(tonnage_data['cutoffs'], tonnage_data['grades'], 'g-', linewidth=2)
                ax2.set_xlabel('Teneur de coupure')
                ax2.set_ylabel('Teneur moyenne')
                ax2.set_title('Teneur moyenne en fonction de la coupure')
                ax2.grid(True)
                
                plt.tight_layout()
                
                # Sauvegarder le graphique dans un fichier temporaire
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                    fig_tonnage.savefig(tmp_file.name, format='png', dpi=150, bbox_inches='tight')
                    tonnage_graph_path = tmp_file.name
                
                # Ajouter le graphique au rapport
                tonnage_img = ReportLabImage(tonnage_graph_path, width=5*inch, height=6*inch)
                elements.append(tonnage_img)
                elements.append(Spacer(1, 0.2*inch))
                
                # Tableau des résultats
                elements.append(Paragraph("4.1 Résultats détaillés", heading2_style))
                
                # Créer un sous-ensemble des résultats (pour ne pas avoir un tableau trop long)
                cutoffs_subset = tonnage_data['cutoffs'][::3]  # Prendre un point sur trois
                tonnages_subset = tonnage_data['tonnages'][::3]
                grades_subset = tonnage_data['grades'][::3]
                metals_subset = tonnage_data['metals'][::3] if 'metals' in tonnage_data else [0] * len(cutoffs_subset)
                
                tonnage_table_data = [["Coupure", "Tonnage (t)", "Teneur moyenne", "Métal contenu"]]
                for i in range(len(cutoffs_subset)):
                    tonnage_table_data.append([
                        cutoffs_subset[i],
                        f"{tonnages_subset[i]:,.0f}",
                        f"{grades_subset[i]:.3f}",
                        f"{metals_subset[i]:,.0f}"
                    ])
                
                tonnage_table = Table(tonnage_table_data, colWidths=[1*inch, 1.5*inch, 1.5*inch, 1.5*inch])
                tonnage_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                elements.append(tonnage_table)
            else:
                # Pour la méthode between, montrer un seul résultat
                min_grade = plot_info.get('min_grade', 0)
                max_grade = plot_info.get('max_grade', 0)
                elements.append(Paragraph(f"Méthode de coupure: Entre {min_grade:.2f} et {max_grade:.2f}", normal_style))
                
                between_data = [
                    ["Paramètre", "Valeur"],
                    ["Tonnage (t)", f"{tonnage_data['tonnages'][0]:,.0f}" if len(tonnage_data['tonnages']) > 0 else "0"],
                    ["Teneur moyenne", f"{tonnage_data['grades'][0]:.3f}" if len(tonnage_data['grades']) > 0 else "0"],
                    ["Métal contenu", f"{tonnage_data['metals'][0]:,.0f}" if 'metals' in tonnage_data and len(tonnage_data['metals']) > 0 else "0"]
                ]
                
                between_table = Table(between_data, colWidths=[2.5*inch, 2*inch])
                between_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
                    ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                elements.append(between_table)
        
        # Conclusion
        elements.append(Paragraph("5. Conclusion", heading1_style))
        elements.append(Paragraph("Ce rapport présente les résultats d'une estimation de ressources minérales "
                                 "par la méthode de l'inverse des distances. Les résultats doivent être interprétés "
                                 "en tenant compte des limitations inhérentes à cette méthode d'estimation.", normal_style))
        elements.append(Spacer(1, 0.1*inch))
        elements.append(Paragraph("L'estimation par inverse des distances est une méthode déterministe qui ne fournit "
                                 "pas d'évaluation directe de l'incertitude associée aux estimations.", normal_style))
        
        # Construire le document PDF
        doc.build(elements)
        
        # Supprimer les fichiers temporaires
        temp_files = [var for var in locals() if var.endswith('_path')]
        for file_path_var in temp_files:
            if os.path.exists(locals()[file_path_var]):
                os.unlink(locals()[file_path_var])
        
        # Récupérer le contenu du buffer
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"Erreur lors de la génération du rapport PDF: {str(e)}")
        return None

# Fonction pour traiter le fichier DXF
def process_dxf_file(dxf_file):
    try:
        # Lire le contenu du fichier dans un buffer
        dxf_content = dxf_file.read()
        file_buffer = io.BytesIO(dxf_content)
        
        # Charger le DXF avec ezdxf
        doc = ezdxf.readfile(file_buffer)
        msp = doc.modelspace()
        
        # Extraire les entités fermées (polylignes, 3DFACE, MESH, etc.)
        entities = []
        
        for entity in msp:
            if entity.dxftype() == 'POLYLINE' or entity.dxftype() == 'LWPOLYLINE':
                if hasattr(entity, 'closed') and entity.closed:
                    vertices = []
                    for vertex in entity.vertices():
                        vertices.append((vertex.dxf.location.x, vertex.dxf.location.y, vertex.dxf.location.z))
                    if len(vertices) >= 3:
                        entities.append({
                            'type': 'polyline',
                            'vertices': vertices
                        })
            elif entity.dxftype() == '3DFACE':
                vertices = [
                    (entity.dxf.vtx0.x, entity.dxf.vtx0.y, entity.dxf.vtx0.z),
                    (entity.dxf.vtx1.x, entity.dxf.vtx1.y, entity.dxf.vtx1.z),
                    (entity.dxf.vtx2.x, entity.dxf.vtx2.y, entity.dxf.vtx2.z),
                    (entity.dxf.vtx3.x, entity.dxf.vtx3.y, entity.dxf.vtx3.z)
                ]
                entities.append({
                    'type': '3dface',
                    'vertices': vertices
                })
            elif entity.dxftype() == 'MESH':
                vertices = []
                for vertex in entity.vertices():
                    vertices.append((vertex.x, vertex.y, vertex.z))
                if len(vertices) >= 3:
                    entities.append({
                        'type': 'mesh',
                        'vertices': vertices
                    })
        
        # Créer un maillage 3D à partir des entités
        mesh_vertices = []
        mesh_faces = []
        for entity in entities:
            vertex_offset = len(mesh_vertices)
            mesh_vertices.extend(entity['vertices'])
            
            if entity['type'] == 'polyline':
                for i in range(1, len(entity['vertices']) - 1):
                    mesh_faces.append([vertex_offset, vertex_offset + i, vertex_offset + i + 1])
            elif entity['type'] == '3dface':
                mesh_faces.append([vertex_offset, vertex_offset + 1, vertex_offset + 2])
                mesh_faces.append([vertex_offset, vertex_offset + 2, vertex_offset + 3])
            elif entity['type'] == 'mesh':
                # Pour un maillage, on suppose qu'il est déjà triangulé
                for i in range(0, len(entity['vertices']), 3):
                    if i + 2 < len(entity['vertices']):
                        mesh_faces.append([vertex_offset + i, vertex_offset + i + 1, vertex_offset + i + 2])
        
        # Créer un maillage trimesh
        if mesh_vertices and mesh_faces:
            mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces)
            
            # Calculer la bounding box
            bounds = mesh.bounds
            min_bounds = {
                'x': bounds[0][0],
                'y': bounds[0][1],
                'z': bounds[0][2]
            }
            max_bounds = {
                'x': bounds[1][0],
                'y': bounds[1][1],
                'z': bounds[1][2]
            }
            
            return {
                'mesh': mesh,
                'bounds': {
                    'min': min_bounds,
                    'max': max_bounds
                },
                'vertices': mesh_vertices,
                'faces': mesh_faces
            }
        else:
            return None
    except Exception as e:
        st.error(f"Erreur lors du traitement du fichier DXF: {str(e)}")
        return None

# Fonction pour créer un lien de téléchargement
def get_download_link(object_to_download, download_filename, download_link_text):
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
        b64 = base64.b64encode(object_to_download.encode()).decode()
        file_type = 'text/csv'
    elif isinstance(object_to_download, plt.Figure):
        buf = BytesIO()
        object_to_download.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        b64 = base64.b64encode(buf.getvalue()).decode()
        file_type = 'image/png'
    elif isinstance(object_to_download, bytes):
        b64 = base64.b64encode(object_to_download).decode()
        file_type = 'application/zip'
    else:
        b64 = base64.b64encode(object_to_download.encode()).decode()
        file_type = 'text/plain'
    
    return f'<a href="data:{file_type};base64,{b64}" download="{download_filename}">{download_link_text}</a>'

# Fonctions pour l'estimation par inverse des distances
def inverse_distance_weighting(point, samples, power, anisotropy):
    if len(samples) == 0:
        return 0
    
    # Si un échantillon est exactement à la position du bloc, retourner sa valeur
    for sample in samples:
        if sample['x'] == point['x'] and sample['y'] == point['y'] and sample['z'] == point['z']:
            return sample['value']
    
    weighted_sum = 0
    weight_sum = 0
    
    for sample in samples:
        # Distance avec anisotropie
        dx = (sample['x'] - point['x']) / anisotropy['x']
        dy = (sample['y'] - point['y']) / anisotropy['y']
        dz = (sample['z'] - point['z']) / anisotropy['z']
        
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        
        if distance == 0:
            return sample['value']  # Si distance nulle, retourner la valeur de l'échantillon
        
        # Calcul du poids par l'inverse de la distance à la puissance p
        weight = 1 / (distance ** power)
        
        weighted_sum += weight * sample['value']
        weight_sum += weight
    
    if weight_sum == 0:
        return 0
    
    return weighted_sum / weight_sum

def is_point_inside_mesh(point, mesh):
    if mesh is None:
        return True
    
    try:
        # Vérification si le point est à l'intérieur du maillage à l'aide de trimesh
        point_array = np.array([point['x'], point['y'], point['z']])
        return bool(mesh.contains([point_array])[0])
    except Exception as e:
        st.warning(f"Erreur lors de la vérification de l'enveloppe: {str(e)}")
        return True  # Par défaut, inclure le point en cas d'erreur

def create_block_model(composites, block_sizes, envelope_data=None, use_envelope=True):
    # Vérifier si les composites existent
    if not composites or len(composites) == 0:
        st.error("Aucun échantillon valide pour créer le modèle de blocs.")
        return [], {'min': {'x': 0, 'y': 0, 'z': 0}, 'max': {'x': 0, 'y': 0, 'z': 0}}
    
    # Déterminer les limites du modèle
    if use_envelope and envelope_data:
        min_bounds = envelope_data['bounds']['min']
        max_bounds = envelope_data['bounds']['max']
    else:
        x_values = [comp['X'] for comp in composites if 'X' in comp]
        y_values = [comp['Y'] for comp in composites if 'Y' in comp]
        z_values = [comp['Z'] for comp in composites if 'Z' in comp]
        
        if not x_values or not y_values or not z_values:
            st.error("Données insuffisantes pour déterminer les limites du modèle.")
            return [], {'min': {'x': 0, 'y': 0, 'z': 0}, 'max': {'x': 0, 'y': 0, 'z': 0}}
        
        min_bounds = {
            'x': math.floor(min(x_values) / block_sizes['x']) * block_sizes['x'],
            'y': math.floor(min(y_values) / block_sizes['y']) * block_sizes['y'],
            'z': math.floor(min(z_values) / block_sizes['z']) * block_sizes['z']
        }
        
        max_bounds = {
            'x': math.ceil(max(x_values) / block_sizes['x']) * block_sizes['x'],
            'y': math.ceil(max(y_values) / block_sizes['y']) * block_sizes['y'],
            'z': math.ceil(max(z_values) / block_sizes['z']) * block_sizes['z']
        }
    
    # Créer les blocs
    blocks = []
    
    x_range = np.arange(min_bounds['x'] + block_sizes['x']/2, max_bounds['x'] + block_sizes['x']/2, block_sizes['x'])
    y_range = np.arange(min_bounds['y'] + block_sizes['y']/2, max_bounds['y'] + block_sizes['y']/2, block_sizes['y'])
    z_range = np.arange(min_bounds['z'] + block_sizes['z']/2, max_bounds['z'] + block_sizes['z']/2, block_sizes['z'])
    
    mesh = envelope_data['mesh'] if envelope_data and use_envelope else None
    
    with st.spinner('Création du modèle de blocs...'):
        progress_bar = st.progress(0)
        total_blocks = len(x_range) * len(y_range) * len(z_range)
        block_count = 0
        
        for x in x_range:
            for y in y_range:
                for z in z_range:
                    block = {
                        'x': x,
                        'y': y,
                        'z': z,
                        'size_x': block_sizes['x'],
                        'size_y': block_sizes['y'],
                        'size_z': block_sizes['z']
                    }
                    
                    # Vérifier si le bloc est dans l'enveloppe
                    if not use_envelope or is_point_inside_mesh(block, mesh):
                        blocks.append(block)
                    
                    block_count += 1
                    if block_count % 100 == 0 or block_count == total_blocks:
                        progress_bar.progress(min(block_count / total_blocks, 1.0))
        
        progress_bar.progress(1.0)
    
    return blocks, {'min': min_bounds, 'max': max_bounds}

def estimate_block_model(empty_blocks, composites, idw_params, search_params, density_method="constant", density_value=2.7):
    estimated_blocks = []
    
    # Vérifier si les entrées sont valides
    if not empty_blocks or len(empty_blocks) == 0:
        st.error("Aucun bloc à estimer.")
        return []
    
    if not composites or len(composites) == 0:
        st.error("Aucun échantillon disponible pour l'estimation.")
        return []
    
    with st.spinner('Estimation en cours...'):
        progress_bar = st.progress(0)
        
        for idx, block in enumerate(stqdm(empty_blocks)):
            progress = idx / len(empty_blocks)
            progress_bar.progress(progress)
            
            # Chercher les échantillons pour l'IDW
            samples = []
            density_samples = []
            
            for composite in composites:
                if 'X' not in composite or 'Y' not in composite or 'Z' not in composite or 'VALUE' not in composite:
                    continue
                
                # Appliquer l'anisotropie
                dx = (composite['X'] - block['x']) / idw_params['anisotropy']['x']
                dy = (composite['Y'] - block['y']) / idw_params['anisotropy']['y']
                dz = (composite['Z'] - block['z']) / idw_params['anisotropy']['z']
                
                distance = math.sqrt(dx**2 + dy**2 + dz**2)
                
                if distance <= max(search_params['x'], search_params['y'], search_params['z']):
                    samples.append({
                        'x': composite['X'],
                        'y': composite['Y'],
                        'z': composite['Z'],
                        'value': composite['VALUE'],
                        'distance': distance
                    })
                    
                    # Si la densité est variable, ajouter les échantillons de densité
                    if density_method == "variable" and 'DENSITY' in composite:
                        density_samples.append({
                            'x': composite['X'],
                            'y': composite['Y'],
                            'z': composite['Z'],
                            'value': composite['DENSITY'],
                            'distance': distance
                        })
            
            samples.sort(key=lambda x: x['distance'])
            
            if len(samples) >= search_params['min_samples']:
                used_samples = samples[:min(len(samples), search_params['max_samples'])]
                
                # Estimation par IDW
                estimate = inverse_distance_weighting(
                    block, 
                    used_samples, 
                    idw_params['power'],
                    idw_params['anisotropy']
                )
                
                estimated_block = block.copy()
                estimated_block['value'] = estimate
                
                # Estimer la densité si nécessaire
                if density_method == "variable" and density_samples:
                    density_samples.sort(key=lambda x: x['distance'])
                    used_density_samples = density_samples[:min(len(density_samples), search_params['max_samples'])]
                    
                    estimated_density = inverse_distance_weighting(
                        block, 
                        used_density_samples, 
                        idw_params['power'],
                        idw_params['anisotropy']
                    )
                    
                    estimated_block['density'] = estimated_density
                else:
                    estimated_block['density'] = density_value
                
                estimated_blocks.append(estimated_block)
        
        progress_bar.progress(1.0)
    
    return estimated_blocks

def calculate_tonnage_grade(blocks, density_method="constant", density_value=2.7, method="above", cutoff_value=None, cutoff_min=None, cutoff_max=None):
    if not blocks:
        return {
            'cutoffs': [],
            'tonnages': [],
            'grades': [],
            'metals': []
        }, {
            'method': method,
            'min_grade': 0,
            'max_grade': 0
        }
    
    # Extraire les valeurs
    values = [block.get('value', 0) for block in blocks]
    
    if not values:
        return {
            'cutoffs': [],
            'tonnages': [],
            'grades': [],
            'metals': []
        }, {
            'method': method,
            'min_grade': 0,
            'max_grade': 0
        }
    
    min_grade = min(values)
    max_grade = max(values)
    
    # Générer les coupures
    step = (max_grade - min_grade) / 20 if max_grade > min_grade else 0.1
    cutoffs = np.arange(min_grade, max_grade + step, max(step, 0.0001))
    
    tonnages = []
    grades = []
    metals = []
    cutoff_labels = []
    
    for cutoff in cutoffs:
        cutoff_labels.append(f"{cutoff:.2f}")
        
        if method == 'above':
            filtered_blocks = [block for block in blocks if block.get('value', 0) >= cutoff]
        elif method == 'below':
            filtered_blocks = [block for block in blocks if block.get('value', 0) <= cutoff]
        elif method == 'between':
            filtered_blocks = [block for block in blocks if cutoff_min <= block.get('value', 0) <= cutoff_max]
            
            # Pour la méthode between, on n'a besoin que d'un seul résultat
            if cutoff > min_grade:
                continue
        
        if not filtered_blocks:
            tonnages.append(0)
            grades.append(0)
            metals.append(0)
            continue
        
        total_tonnage = 0
        total_metal = 0
        
        for block in filtered_blocks:
            if 'size_x' in block and 'size_y' in block and 'size_z' in block:
                block_volume = block['size_x'] * block['size_y'] * block['size_z']
                block_density = block.get('density', density_value) if density_method == "variable" else density_value
                block_tonnage = block_volume * block_density
                
                total_tonnage += block_tonnage
                total_metal += block_tonnage * block.get('value', 0)
        
        if total_tonnage > 0:
            avg_grade = total_metal / total_tonnage
        else:
            avg_grade = 0
        
        tonnages.append(total_tonnage)
        grades.append(avg_grade)
        metals.append(total_metal)
    
    return {
        'cutoffs': cutoff_labels,
        'tonnages': tonnages,
        'grades': grades,
        'metals': metals
    }, {
        'method': method,
        'min_grade': min_grade,
        'max_grade': max_grade
    }

# Fonctions de visualisation
def plot_3d_model_with_cubes(blocks, composites, envelope_data=None, block_scale=0.9):
    fig = go.Figure()
    
    # Ajouter les composites
    if composites:
        x = [comp.get('X', 0) for comp in composites if 'X' in comp and 'Y' in comp and 'Z' in comp and 'VALUE' in comp]
        y = [comp.get('Y', 0) for comp in composites if 'X' in comp and 'Y' in comp and 'Z' in comp and 'VALUE' in comp]
        z = [comp.get('Z', 0) for comp in composites if 'X' in comp and 'Y' in comp and 'Z' in comp and 'VALUE' in comp]
        values = [comp.get('VALUE', 0) for comp in composites if 'X' in comp and 'Y' in comp and 'Z' in comp and 'VALUE' in comp]
        
        if x and y and z and values:
            composite_scatter = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(
                    size=3,
                    color=values,
                    colorscale='Viridis',
                    opacity=0.8,
                    colorbar=dict(title="Teneur")
                ),
                text=[f"Teneur: {v:.3f}" for v in values],
                name='Composites'
            )
            fig.add_trace(composite_scatter)
    
    # Ajouter les blocs en tant que cubes
    if blocks:
        # Vérifier les clés nécessaires dans les blocs
        valid_blocks = [block for block in blocks 
                      if 'x' in block and 'y' in block and 'z' in block 
                      and 'size_x' in block and 'size_y' in block and 'size_z' in block 
                      and 'value' in block]
        
        if not valid_blocks:
            st.warning("Aucun bloc valide à afficher.")
            return fig
        
        # Limiter le nombre de blocs pour éviter de surcharger la visualisation
        max_display_blocks = 2000
        if len(valid_blocks) > max_display_blocks:
            st.warning(f"Le modèle contient {len(valid_blocks)} blocs. Pour une meilleure performance, seuls {max_display_blocks} blocs sont affichés.")
            valid_blocks = valid_blocks[:max_display_blocks]
        
        # Créer des cubes pour chaque bloc (en utilisant Mesh3d)
        x_vals = []
        y_vals = []
        z_vals = []
        i_vals = []
        j_vals = []
        k_vals = []
        intensity = []
        
        for idx, block in enumerate(valid_blocks):
            # Créer les 8 sommets d'un cube
            x_size = block['size_x'] * block_scale / 2
            y_size = block['size_y'] * block_scale / 2
            z_size = block['size_z'] * block_scale / 2
            
            x0, y0, z0 = block['x'] - x_size, block['y'] - y_size, block['z'] - z_size
            x1, y1, z1 = block['x'] + x_size, block['y'] + y_size, block['z'] + z_size
            
            # Ajouter les sommets
            vertices = [
                (x0, y0, z0),  # 0
                (x1, y0, z0),  # 1
                (x1, y1, z0),  # 2
                (x0, y1, z0),  # 3
                (x0, y0, z1),  # 4
                (x1, y0, z1),  # 5
                (x1, y1, z1),  # 6
                (x0, y1, z1)   # 7
            ]
            
            # Ajouter les faces du cube (triangles)
            faces = [
                (0, 1, 2), (0, 2, 3),  # bottom
                (4, 5, 6), (4, 6, 7),  # top
                (0, 1, 5), (0, 5, 4),  # front
                (2, 3, 7), (2, 7, 6),  # back
                (0, 3, 7), (0, 7, 4),  # left
                (1, 2, 6), (1, 6, 5)   # right
            ]
            
            for v in vertices:
                x_vals.append(v[0])
                y_vals.append(v[1])
                z_vals.append(v[2])
                intensity.append(block['value'])
            
            offset = idx * 8  # 8 sommets par cube
            for f in faces:
                i_vals.append(offset + f[0])
                j_vals.append(offset + f[1])
                k_vals.append(offset + f[2])
        
        if x_vals and y_vals and z_vals and i_vals and j_vals and k_vals and intensity:
            block_mesh = go.Mesh3d(
                x=x_vals,
                y=y_vals,
                z=z_vals,
                i=i_vals,
                j=j_vals,
                k=k_vals,
                intensity=intensity,
                colorscale='Viridis',
                opacity=0.7,
                name='Blocs estimés',
                colorbar=dict(title="Teneur")
            )
            fig.add_trace(block_mesh)
    
    # Ajouter l'enveloppe DXF
    if envelope_data and 'vertices' in envelope_data and 'faces' in envelope_data:
        vertices = envelope_data['vertices']
        faces = envelope_data['faces']
        
        if vertices and faces:
            i, j, k = [], [], []
            for face in faces:
                if len(face) >= 3:
                    i.append(face[0])
                    j.append(face[1])
                    k.append(face[2])
            
            if i and j and k:
                wireframe = go.Mesh3d(
                    x=[v[0] for v in vertices],
                    y=[v[1] for v in vertices],
                    z=[v[2] for v in vertices],
                    i=i, j=j, k=k,
                    opacity=0.3,
                    color='green',
                    name='Enveloppe'
                )
                fig.add_trace(wireframe)
    
    # Mise en page
    fig.update_layout(
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectratio=dict(x=1, y=1, z=1)
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        legend=dict(x=0, y=0.9)
    )
    
    return fig

def plot_histogram(values, title, color='steelblue'):
    if not values or len(values) <= 1:
        # Créer un graphique vide avec un message
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Données insuffisantes pour l'histogramme", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title(title)
        return fig
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculer le nombre de bins (au moins 5)
    n_bins = max(5, int(1 + 3.322 * math.log10(len(values))))
    
    sns.histplot(values, bins=n_bins, kde=True, color=color, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Valeur')
    ax.set_ylabel('Fréquence')
    
    return fig

def plot_tonnage_grade(tonnage_data, plot_info=None):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Vérifier si les données sont valides
    if (not tonnage_data or 'cutoffs' not in tonnage_data or 'tonnages' not in tonnage_data or 'grades' not in tonnage_data or
        len(tonnage_data['cutoffs']) == 0 or len(tonnage_data['tonnages']) == 0 or len(tonnage_data['grades']) == 0):
        # Ajouter un texte indiquant l'absence de données
        fig.add_annotation(
            x=0.5, y=0.5,
            text="Données insuffisantes pour le graphique tonnage-teneur",
            showarrow=False,
            font=dict(size=14)
        )
        return fig
    
    if plot_info and plot_info.get('method') == 'between':
        # Pour la méthode 'between', on utilise un graphique à barres
        fig.add_trace(
            go.Bar(
                x=['Résultat'],
                y=[tonnage_data['tonnages'][0]],
                name='Tonnage',
                marker_color='rgb(63, 81, 181)'
            )
        )
        
        fig.add_trace(
            go.Bar(
                x=['Résultat'],
                y=[tonnage_data['grades'][0]],
                name='Teneur moyenne',
                marker_color='rgb(0, 188, 212)'
            ),
            secondary_y=True
        )
    else:
        # Pour les méthodes 'above' et 'below', on utilise un graphique en ligne
        fig.add_trace(
            go.Scatter(
                x=tonnage_data['cutoffs'],
                y=tonnage_data['tonnages'],
                name='Tonnage',
                fill='tozeroy',
                mode='lines',
                line=dict(color='rgb(63, 81, 181)')
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=tonnage_data['cutoffs'],
                y=tonnage_data['grades'],
                name='Teneur moyenne',
                mode='lines',
                line=dict(color='rgb(0, 188, 212)')
            ),
            secondary_y=True
        )
    
    fig.update_layout(
        title_text='Courbe Tonnage-Teneur',
        xaxis_title='Teneur de coupure',
        legend=dict(x=0, y=1.1, orientation='h')
    )
    
    fig.update_yaxes(title_text='Tonnage (t)', secondary_y=False)
    fig.update_yaxes(title_text='Teneur moyenne', secondary_y=True)
    
    return fig

def plot_metal_content(tonnage_data, plot_info=None):
    fig = go.Figure()
    
    # Vérifier si les données sont valides
    if (not tonnage_data or 'cutoffs' not in tonnage_data or 'metals' not in tonnage_data or
        len(tonnage_data['cutoffs']) == 0 or len(tonnage_data['metals']) == 0):
        # Ajouter un texte indiquant l'absence de données
        fig.add_annotation(
            x=0.5, y=0.5,
            text="Données insuffisantes pour le graphique de métal contenu",
            showarrow=False,
            font=dict(size=14)
        )
        return fig
    
    if plot_info and plot_info.get('method') == 'between':
        # Pour la méthode 'between', on utilise un graphique à barres
        if len(tonnage_data['metals']) > 0:
            fig.add_trace(
                go.Bar(
                    x=['Résultat'],
                    y=[tonnage_data['metals'][0]],
                    name='Métal contenu',
                    marker_color='rgb(76, 175, 80)'
                )
            )
    else:
        # Pour les méthodes 'above' et 'below', on utilise un graphique en ligne
        fig.add_trace(
            go.Scatter(
                x=tonnage_data['cutoffs'],
                y=tonnage_data['metals'],
                name='Métal contenu',
                fill='tozeroy',
                mode='lines',
                line=dict(color='rgb(76, 175, 80)')
            )
        )
    
    fig.update_layout(
        title_text='Métal contenu',
        xaxis_title='Teneur de coupure',
        yaxis_title='Métal contenu'
    )
    
    return fig

# Interface utilisateur Streamlit
# Logo en haut de la page
logo = create_mining_logo()
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.image(logo, width=150)
    st.title("MineEstim - Estimation par inverse des distances")
    st.caption("Développé par Didier Ouedraogo, P.Geo")

# Sidebar - Chargement des données et paramètres
with st.sidebar:
    st.header("Données")
    
    # Nom du projet
    project_name = st.text_input("Nom du projet", "Projet Cuivre")
    
    # Option pour générer des données d'exemple
    use_example_data = st.checkbox("Utiliser des données d'exemple (gisement de cuivre)", value=False)
    
    if use_example_data:
        n_samples = st.slider("Nombre d'échantillons", min_value=100, max_value=1000, value=300, step=50)
        df = generate_synthetic_copper_data(n_samples)
        st.success(f"{len(df)} échantillons générés pour un gisement de cuivre synthétique")
        uploaded_file = True  # Simuler un fichier chargé
    else:
        uploaded_file = st.file_uploader("Fichier CSV des composites", type=["csv"])
    
    if uploaded_file is True or uploaded_file:  # Check for both boolean and file object
        if not use_example_data:
            try:
                # Section ajoutée pour la conversion CSV améliorée
                st.write("### Options de lecture CSV")
                
                try_encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
                selected_encoding = st.selectbox(
                    "Encodage du fichier", 
                    options=try_encodings,
                    index=0,
                    help="Si vous voyez des caractères bizarres, essayez un autre encodage"
                )
                
                separator_options = [',', ';', '\t', '|']
                selected_separator = st.selectbox(
                    "Séparateur", 
                    options=separator_options, 
                    index=0,
                    format_func=lambda x: "Virgule (,)" if x == ',' else "Point-virgule (;)" if x == ';' else "Tabulation" if x == '\t' else "Pipe (|)"
                )
                
                decimal_options = ['.', ',']
                selected_decimal = st.selectbox(
                    "Séparateur décimal", 
                    options=decimal_options,
                    index=0,
                    format_func=lambda x: "Point (.)" if x == '.' else "Virgule (,)"
                )
                
                # Charge le fichier avec les options sélectionnées
                df = pd.read_csv(
                    uploaded_file, 
                    sep=selected_separator, 
                    decimal=selected_decimal, 
                    encoding=selected_encoding,
                    low_memory=False,
                    on_bad_lines='warn'
                )
                
                st.success(f"{len(df)} lignes chargées")
                
                # Convertir automatiquement les colonnes qui semblent être numériques
                for col in df.columns:
                    try:
                        # Check if column has non-numeric characters in what should be numbers
                        if df[col].dtype == 'object':
                            # Try to convert to numeric, ignore errors
                            numeric_values = pd.to_numeric(df[col], errors='coerce')
                            # If at least 80% of values can be converted, do the conversion
                            if numeric_values.notna().sum() / len(df) > 0.8:
                                df[col] = numeric_values
                    except:
                        pass  # Skip if any error
                
                # Afficher un aperçu des données brutes pour diagnostiquer
                with st.expander("Aperçu des données chargées", expanded=True):
                    st.write("Aperçu des premières lignes :", df.head())
                    st.write("Types de données :", df.dtypes)
                    st.write("Nombre de valeurs non-nulles :", df.count())
                
            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier: {str(e)}")
                st.code(traceback.format_exc())
                st.stop()
        
        # Vérifier si le DataFrame est valide
        if df is None or df.empty:
            st.error("Le fichier CSV ne contient pas de données valides.")
            st.stop()
        
        # Mappage des colonnes
        st.subheader("Mappage des colonnes")
        
        try:
            # Utiliser un index sécurisé en cas d'absence de colonnes
            col_x = st.selectbox("Colonne X", options=df.columns, 
                                index=df.columns.get_loc('X') if 'X' in df.columns else 0)
            col_y = st.selectbox("Colonne Y", options=df.columns, 
                                index=df.columns.get_loc('Y') if 'Y' in df.columns else 0)
            col_z = st.selectbox("Colonne Z", options=df.columns, 
                                index=df.columns.get_loc('Z') if 'Z' in df.columns else 0)
            
            # Pour un gisement de cuivre, utiliser CU_PCT comme colonne de teneur par défaut
            value_col_index = (df.columns.get_loc('CU_PCT') if 'CU_PCT' in df.columns else 
                            (df.columns.get_loc('VALUE') if 'VALUE' in df.columns else 0))
            col_value = st.selectbox("Colonne Teneur", options=df.columns, index=value_col_index)
        except Exception as e:
            st.error(f"Erreur lors du mappage des colonnes: {str(e)}")
            st.stop()
        
        # Option pour la densité
        density_options = ["Constante", "Variable (colonne)"]
        density_method = st.radio("Méthode de densité", options=density_options)
        
        if density_method == "Constante":
            density_value = st.number_input("Densité (t/m³)", min_value=0.1, value=2.7, step=0.1)
            density_column = None
        else:
            density_column = st.selectbox("Colonne Densité", options=df.columns, 
                                        index=df.columns.get_loc('DENSITY') if 'DENSITY' in df.columns else 0)
            if density_column in df.columns:
                st.info(f"Densité moyenne des échantillons: {df[density_column].mean():.2f} t/m³")
        
        # Filtres optionnels
        st.subheader("Filtre (facultatif)")
        
        domain_options = ['-- Aucun --'] + list(df.columns)
        domain_index = domain_options.index('DOMAIN') if 'DOMAIN' in domain_options else 0
        col_domain = st.selectbox("Colonne de domaine", options=domain_options, index=domain_index)
        
        # Si un domaine est sélectionné
        domain_filter_value = None
        if col_domain != '-- Aucun --':
            domain_filter_type = st.selectbox("Type de filtre", options=["=", "!=", "IN", "NOT IN"])
            
            if domain_filter_type in ["=", "!="]:
                unique_values = df[col_domain].dropna().unique()
                if len(unique_values) > 0:
                    # Utiliser une liste déroulante pour sélectionner une valeur unique
                    domain_filter_value = st.selectbox("Valeur", options=unique_values)
                else:
                    domain_filter_value = st.text_input("Valeur")
            else:
                domain_values = df[col_domain].dropna().unique()
                domain_filter_value = st.multiselect("Valeurs", options=domain_values)
        
        # Enveloppe DXF
        st.subheader("Enveloppe (facultatif)")
        
        envelope_method = st.radio("Méthode d'enveloppe", ["Manuelle", "DXF"])
        
        if envelope_method == "DXF":
            uploaded_dxf = st.file_uploader("Fichier DXF de l'enveloppe", type=["dxf"])
            
            if uploaded_dxf:
                envelope_data = process_dxf_file(uploaded_dxf)
                if envelope_data:
                    st.success(f"Enveloppe DXF chargée avec succès")
                    st.session_state.envelope_data = envelope_data
                else:
                    st.error("Impossible de traiter le fichier DXF. Assurez-vous qu'il contient des entités 3D fermées.")
                    st.session_state.envelope_data = None
        else:  # Enveloppe manuelle
            try:
                col1, col2 = st.columns(2)
                
                # Valeurs par défaut sécurisées pour les limites min/max
                default_min_x = float(df[col_x].min()) if pd.api.types.is_numeric_dtype(df[col_x]) else 0
                default_min_y = float(df[col_y].min()) if pd.api.types.is_numeric_dtype(df[col_y]) else 0
                default_min_z = float(df[col_z].min()) if pd.api.types.is_numeric_dtype(df[col_z]) else 0
                default_max_x = float(df[col_x].max()) if pd.api.types.is_numeric_dtype(df[col_x]) else 100
                default_max_y = float(df[col_y].max()) if pd.api.types.is_numeric_dtype(df[col_y]) else 100
                default_max_z = float(df[col_z].max()) if pd.api.types.is_numeric_dtype(df[col_z]) else 100
                
                with col1:
                    st.markdown("Minimum")
                    min_x = st.number_input("Min X", value=default_min_x, format="%.2f")
                    min_y = st.number_input("Min Y", value=default_min_y, format="%.2f")
                    min_z = st.number_input("Min Z", value=default_min_z, format="%.2f")
                
                with col2:
                    st.markdown("Maximum")
                    max_x = st.number_input("Max X", value=default_max_x, format="%.2f")
                    max_y = st.number_input("Max Y", value=default_max_y, format="%.2f")
                    max_z = st.number_input("Max Z", value=default_max_z, format="%.2f")
                
                envelope_bounds = {
                    'min': {'x': min_x, 'y': min_y, 'z': min_z},
                    'max': {'x': max_x, 'y': max_y, 'z': max_z}
                }
                
                # Créer une enveloppe simple à partir des limites manuelles
                if envelope_method == "Manuelle":
                    # Créer un cube simple à partir des limites
                    vertices = [
                        (min_x, min_y, min_z),
                        (max_x, min_y, min_z),
                        (max_x, max_y, min_z),
                        (min_x, max_y, min_z),
                        (min_x, min_y, max_z),
                        (max_x, min_y, max_z),
                        (max_x, max_y, max_z),
                        (min_x, max_y, max_z)
                    ]
                    
                    # Définir les faces du cube
                    faces = [
                        [0, 1, 2], [0, 2, 3],  # bottom
                        [4, 5, 6], [4, 6, 7],  # top
                        [0, 1, 5], [0, 5, 4],  # front
                        [2, 3, 7], [2, 7, 6],  # back
                        [0, 3, 7], [0, 7, 4],  # left
                        [1, 2, 6], [1, 6, 5]   # right
                    ]
                    
                    # Créer un maillage trimesh
                    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                    
                    # Stocker dans la session
                    st.session_state.envelope_data = {
                        'mesh': mesh,
                        'bounds': envelope_bounds,
                        'vertices': vertices,
                        'faces': faces
                    }
            except Exception as e:
                st.error(f"Erreur lors de la création de l'enveloppe manuelle: {str(e)}")
                st.session_state.envelope_data = None
        
        use_envelope = st.checkbox("Restreindre l'estimation à l'enveloppe", value=True)
        st.session_state.use_envelope = use_envelope
    
    # Paramètres IDW
    st.header("Paramètres IDW")
    
    power = st.slider("Puissance (p)", min_value=1.0, max_value=5.0, value=2.0, step=0.1,
                    help="Plus la valeur est élevée, plus l'influence des points proches est grande")
    
    st.subheader("Anisotropie (ratio des distances)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        anisotropy_x = st.number_input("X", min_value=0.1, value=1.0, step=0.1)
    
    with col2:
        anisotropy_y = st.number_input("Y", min_value=0.1, value=1.0, step=0.1)
    
    with col3:
        anisotropy_z = st.number_input("Z", min_value=0.1, value=0.5, step=0.1)
    
    # Paramètres du modèle de blocs
    st.header("Paramètres du modèle")
    
    st.subheader("Taille des blocs (m)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        block_size_x = st.number_input("X", min_value=1, value=10, step=1)
    
    with col2:
        block_size_y = st.number_input("Y", min_value=1, value=10, step=1)
    
    with col3:
        block_size_z = st.number_input("Z", min_value=1, value=5, step=1)
    
    st.subheader("Rayon de recherche (m)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_radius_x = st.number_input("X ", min_value=1, value=50, step=1)
    
    with col2:
        search_radius_y = st.number_input("Y ", min_value=1, value=50, step=1)
    
    with col3:
        search_radius_z = st.number_input("Z ", min_value=1, value=25, step=1)
    
    min_samples = st.number_input("Nombre min d'échantillons", min_value=1, value=2, step=1)
    max_samples = st.number_input("Nombre max d'échantillons", min_value=1, value=10, step=1)

# Traitement des données
if uploaded_file is True or uploaded_file:  # Check for both boolean and file object
    # Diagnostic complet des données - affiché directement dans la page principale
    st.subheader("Diagnostic des données")
    
    diagnostic_col1, diagnostic_col2 = st.columns(2)
    
    with diagnostic_col1:
        st.write("### Informations sur le DataFrame")
        st.write(f"Forme du DataFrame : {df.shape}")
        st.write(f"Types de données :")
        st.write(df.dtypes)
        
        # Afficher les informations sur les colonnes mappées
        st.write("### Colonnes sélectionnées")
        st.write(f"Colonne X : '{col_x}', Type : {df[col_x].dtype}")
        st.write(f"Colonne Y : '{col_y}', Type : {df[col_y].dtype}")
        st.write(f"Colonne Z : '{col_z}', Type : {df[col_z].dtype}")
        st.write(f"Colonne Teneur : '{col_value}', Type : {df[col_value].dtype}")
        
        # Vérifier si les colonnes mappées existent
        missing_cols = []
        for col in [col_x, col_y, col_z, col_value]:
            if col not in df.columns:
                missing_cols.append(col)
        
        if missing_cols:
            st.error(f"Les colonnes suivantes n'existent pas dans le DataFrame : {', '.join(missing_cols)}")
            st.stop()
    
    with diagnostic_col2:
        st.write("### Analyse des valeurs manquantes")
        na_info = pd.DataFrame({
            'Colonne': [col_x, col_y, col_z, col_value],
            'Valeurs manquantes': [
                df[col_x].isna().sum(),
                df[col_y].isna().sum(),
                df[col_z].isna().sum(),
                df[col_value].isna().sum()
            ],
            'Pourcentage manquant': [
                f"{df[col_x].isna().mean() * 100:.1f}%",
                f"{df[col_y].isna().mean() * 100:.1f}%",
                f"{df[col_z].isna().mean() * 100:.1f}%",
                f"{df[col_value].isna().mean() * 100:.1f}%"
            ]
        })
        st.write(na_info)
        
        # Vérifier les valeurs non numériques
        non_numeric = pd.DataFrame({
            'Colonne': [col_x, col_y, col_z, col_value],
            'Valeurs non numériques': [
                pd.to_numeric(df[col_x], errors='coerce').isna().sum() - df[col_x].isna().sum(),
                pd.to_numeric(df[col_y], errors='coerce').isna().sum() - df[col_y].isna().sum(),
                pd.to_numeric(df[col_z], errors='coerce').isna().sum() - df[col_z].isna().sum(),
                pd.to_numeric(df[col_value], errors='coerce').isna().sum() - df[col_value].isna().sum()
            ]
        })
        st.write("### Analyse des valeurs non numériques")
        st.write(non_numeric)
    
    # Afficher un échantillon de données pour vérifier le format
    with st.expander("Aperçu des données", expanded=False):
        st.write("### Échantillon de données")
        st.write(df.head())
    
    # Appliquer le filtre de domaine si nécessaire
    filtered_df = df.copy()
    
    if col_domain != '-- Aucun --' and domain_filter_value is not None:
        st.write(f"### Application du filtre de domaine : {col_domain} {domain_filter_type} {domain_filter_value}")
        before_filter = len(filtered_df)
        
        try:
            if domain_filter_type == "=":
                filtered_df = filtered_df[filtered_df[col_domain] == domain_filter_value]
            elif domain_filter_type == "!=":
                filtered_df = filtered_df[filtered_df[col_domain] != domain_filter_value]
            elif domain_filter_type == "IN":
                filtered_df = filtered_df[filtered_df[col_domain].isin(domain_filter_value)]
            elif domain_filter_type == "NOT IN":
                filtered_df = filtered_df[~filtered_df[col_domain].isin(domain_filter_value)]
            
            after_filter = len(filtered_df)
            st.write(f"Lignes avant filtre : {before_filter}")
            st.write(f"Lignes après filtre : {after_filter}")
            st.write(f"Lignes retirées : {before_filter - after_filter}")
            
            if after_filter == 0:
                st.error("Le filtre a supprimé toutes les lignes ! Vérifiez vos critères de filtrage.")
                st.stop()
        except Exception as e:
            st.error(f"Erreur lors de l'application du filtre : {str(e)}")
            st.write("Vérifiez que le type de données de la colonne de domaine est compatible avec votre filtre.")
            st.stop()
    
    # Conversion forcée en numérique avec diagnostic approfondi
    st.write("### Conversion des données en format numérique")
    
    # Créer des copies des colonnes converties en numérique
    try:
        numeric_cols = {}
        for col in [col_x, col_y, col_z, col_value]:
            # Tentative de conversion avec gestion des erreurs
            numeric_cols[col] = pd.to_numeric(filtered_df[col], errors='coerce')
            
            # Afficher les statistiques de conversion
            na_before = filtered_df[col].isna().sum()
            na_after = numeric_cols[col].isna().sum()
            new_na = na_after - na_before
            
            st.write(f"Colonne '{col}' : {new_na} valeurs non convertibles en nombres")
            
            if new_na > 0:
                # Montrer quelques exemples de valeurs problématiques
                problem_values = filtered_df.loc[numeric_cols[col].isna() & ~filtered_df[col].isna(), col].head(5).tolist()
                st.write(f"Exemples de valeurs problématiques : {problem_values}")
        
        # Créer un masque pour les lignes valides après conversion
        valid_mask = (
            numeric_cols[col_x].notna() & 
            numeric_cols[col_y].notna() & 
            numeric_cols[col_z].notna() & 
            numeric_cols[col_value].notna()
        )
        
        # Calculer le nombre de lignes valides
        valid_count = valid_mask.sum()
        st.write(f"Nombre de lignes avec toutes les valeurs numériques valides : {valid_count}")
        
        if valid_count == 0:
            st.error("Aucune ligne ne contient toutes les valeurs numériques requises !")
            st.write("Vérifiez le format des données et assurez-vous que toutes les colonnes contiennent des nombres valides.")
            st.stop()
        
        # Préparation des composites avec mappage des colonnes
        composites_data = []
        
        # Créer la liste des composites à partir des lignes valides
        for idx, row in filtered_df[valid_mask].iterrows():
            composite = {
                'X': float(numeric_cols[col_x][idx]),
                'Y': float(numeric_cols[col_y][idx]),
                'Z': float(numeric_cols[col_z][idx]),
                'VALUE': float(numeric_cols[col_value][idx]),
                'DOMAIN': row[col_domain] if col_domain != '-- Aucun --' and col_domain in row else None
            }
            
            # Ajouter la densité si elle est variable
            if density_method == "Variable (colonne)" and density_column and density_column in row:
                try:
                    density_val = float(pd.to_numeric(row[density_column], errors='coerce'))
                    if not pd.isna(density_val):
                        composite['DENSITY'] = density_val
                except (ValueError, TypeError):
                    pass  # Ignorer les erreurs de conversion pour la densité
            
            composites_data.append(composite)
        
        # Vérifier si des composites ont été créés
        st.write(f"### Résultat : {len(composites_data)} composites valides créés")
        
        if not composites_data:
            st.error("Aucun échantillon valide après filtrage et conversion. Vérifiez vos filtres et mappages de colonnes.")
            st.stop()
        else:
            # Afficher un aperçu des composites créés
            st.write("### Aperçu des composites créés")
            composites_preview = pd.DataFrame(composites_data[:5])
            st.write(composites_preview)
            
            # Si tout va bien, afficher un message de succès et continuer
            st.success(f"Traitement réussi : {len(composites_data)} composites créés et prêts pour l'estimation.")
    
    except Exception as e:
        st.error(f"Erreur lors du traitement des données : {str(e)}")
        st.write("Détails de l'erreur pour le débogage :")
        st.exception(e)
        st.stop()

    # Afficher les statistiques des composites
    composite_values = [comp['VALUE'] for comp in composites_data]
    composite_stats = calculate_stats(composite_values)
    
    # Onglets principaux
    tabs = st.tabs(["Modèle 3D", "Statistiques", "Tonnage-Teneur", "Rapport"])
    
    with tabs[0]:  # Modèle 3D
        st.subheader("Modèle de blocs 3D")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            create_model_button = st.button("Créer le modèle de blocs", type="primary")
            
            if "empty_blocks" in st.session_state and st.session_state.empty_blocks:
                estimate_button = st.button("Estimer par IDW", type="primary")
            
            # Options d'affichage
            st.subheader("Options d'affichage")
            show_composites = st.checkbox("Afficher les composites", value=True)
            show_blocks = st.checkbox("Afficher les blocs", value=True)
            show_envelope = st.checkbox("Afficher l'enveloppe", value=True if 'envelope_data' in st.session_state and st.session_state.envelope_data else False)
            
            # Taille des cubes
            block_scale = st.slider("Taille des blocs (échelle)", min_value=0.1, max_value=1.0, value=0.9, step=0.05)
        
        with col1:
            try:
                if create_model_button:
                    # Créer le modèle de blocs vide
                    block_sizes = {'x': block_size_x, 'y': block_size_y, 'z': block_size_z}
                    envelope_data = st.session_state.envelope_data if 'envelope_data' in st.session_state else None
                    use_envelope = st.session_state.use_envelope if 'use_envelope' in st.session_state else False
                    
                    empty_blocks, model_bounds = create_block_model(
                        composites_data, 
                        block_sizes, 
                        envelope_data, 
                        use_envelope
                    )
                    
                    if not empty_blocks:
                        st.error("Impossible de créer le modèle de blocs. Vérifiez vos paramètres et données.")
                    else:
                        st.session_state.empty_blocks = empty_blocks
                        st.session_state.model_bounds = model_bounds
                        
                        st.success(f"Modèle créé avec {len(empty_blocks)} blocs")
                        
                        # Afficher le modèle 3D
                        envelope_data_to_show = envelope_data if show_envelope and envelope_data else None
                        fig = plot_3d_model_with_cubes(
                            [],
                            composites_data if show_composites else [],
                            envelope_data_to_show,
                            block_scale
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                elif "empty_blocks" in st.session_state and estimate_button:
                    # Paramètres pour l'IDW
                    idw_params = {
                        'power': power,
                        'anisotropy': {'x': anisotropy_x, 'y': anisotropy_y, 'z': anisotropy_z}
                    }
                    
                    search_params = {
                        'x': search_radius_x,
                        'y': search_radius_y,
                        'z': search_radius_z,
                        'min_samples': min_samples,
                        'max_samples': max_samples
                    }
                    
                    # Détermine la méthode de densité
                    if density_method == "Variable (colonne)":
                        density_method_str = "variable"
                        density_value_num = None
                    else:
                        density_method_str = "constant"
                        density_value_num = density_value
                    
                    # Estimer le modèle
                    estimated_blocks = estimate_block_model(
                        st.session_state.empty_blocks, 
                        composites_data, 
                        idw_params, 
                        search_params,
                        density_method_str,
                        density_value_num
                    )
                    
                    if not estimated_blocks:
                        st.error("L'estimation n'a pas produit de blocs. Vérifiez vos paramètres.")
                    else:
                        st.session_state.estimated_blocks = estimated_blocks
                        
                        # Stocker les paramètres pour le rapport
                        st.session_state.idw_params = idw_params
                        st.session_state.search_params = search_params
                        st.session_state.block_sizes = {'x': block_size_x, 'y': block_size_y, 'z': block_size_z}
                        st.session_state.density_method = density_method_str
                        st.session_state.density_value = density_value_num if density_method_str == "constant" else None
                        st.session_state.density_column = density_column if density_method_str == "variable" else None
                        st.session_state.project_name = project_name
                        
                        st.success(f"Estimation terminée, {len(estimated_blocks)} blocs estimés")
                        
                        # Afficher le modèle estimé
                        envelope_data_to_show = st.session_state.envelope_data if show_envelope and 'envelope_data' in st.session_state and st.session_state.envelope_data else None
                        fig = plot_3d_model_with_cubes(
                            estimated_blocks if show_blocks else [],
                            composites_data if show_composites else [],
                            envelope_data_to_show,
                            block_scale
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Section d'export
                        st.subheader("Exporter")
                        
                        # Export du modèle de blocs en CSV
                        if st.button("Exporter modèle de blocs (CSV)"):
                            # Créer un DataFrame pour l'export
                            export_df = pd.DataFrame(estimated_blocks)
                            
                            # Renommer les colonnes pour correspondre au format d'origine
                            export_df = export_df.rename(columns={
                                'x': 'X', 'y': 'Y', 'z': 'Z', 'value': 'VALUE',
                                'size_x': 'SIZE_X', 'size_y': 'SIZE_Y', 'size_z': 'SIZE_Z',
                                'density': 'DENSITY'
                            })
                            
                            # Ajouter des informations supplémentaires
                            export_df['VOLUME'] = export_df['SIZE_X'] * export_df['SIZE_Y'] * export_df['SIZE_Z']
                            export_df['TONNAGE'] = export_df['VOLUME'] * export_df['DENSITY']
                            export_df['METAL_CONTENT'] = export_df['VALUE'] * export_df['TONNAGE']
                            
                            # Créer le lien de téléchargement
                            csv = export_df.to_csv(index=False)
                            st.download_button(
                                label="Télécharger CSV",
                                data=csv,
                                file_name=f"{project_name.replace(' ', '_')}_modele_blocs_idw.csv",
                                mime="text/csv"
                            )
                
                elif "estimated_blocks" in st.session_state:
                    # Afficher le modèle estimé déjà calculé
                    envelope_data_to_show = st.session_state.envelope_data if show_envelope and 'envelope_data' in st.session_state and st.session_state.envelope_data else None
                    fig = plot_3d_model_with_cubes(
                        st.session_state.estimated_blocks if show_blocks else [],
                        composites_data if show_composites else [],
                        envelope_data_to_show,
                        block_scale
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Section d'export
                    st.subheader("Exporter")
                    
                    # Export du modèle de blocs en CSV
                    if st.button("Exporter modèle de blocs (CSV)"):
                        # Créer un DataFrame pour l'export
                        export_df = pd.DataFrame(st.session_state.estimated_blocks)
                        
                        # Renommer les colonnes pour correspondre au format d'origine
                        export_df = export_df.rename(columns={
                            'x': 'X', 'y': 'Y', 'z': 'Z', 'value': 'VALUE',
                            'size_x': 'SIZE_X', 'size_y': 'SIZE_Y', 'size_z': 'SIZE_Z',
                            'density': 'DENSITY'
                        })
                        
                        # Ajouter des informations supplémentaires
                        export_df['VOLUME'] = export_df['SIZE_X'] * export_df['SIZE_Y'] * export_df['SIZE_Z']
                        export_df['TONNAGE'] = export_df['VOLUME'] * export_df['DENSITY']
                        export_df['METAL_CONTENT'] = export_df['VALUE'] * export_df['TONNAGE']
                        
                        # Créer le lien de téléchargement
                        csv = export_df.to_csv(index=False)
                        st.download_button(
                            label="Télécharger CSV",
                            data=csv,
                            file_name=f"{project_name.replace(' ', '_')}_modele_blocs_idw.csv",
                            mime="text/csv"
                        )
                
                else:
                    # Afficher seulement les composites si aucun modèle n'est créé
                    envelope_data_to_show = st.session_state.envelope_data if show_envelope and 'envelope_data' in st.session_state and st.session_state.envelope_data else None
                    fig = plot_3d_model_with_cubes(
                        [],
                        composites_data if show_composites else [],
                        envelope_data_to_show,
                        block_scale
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Erreur dans l'onglet Modèle 3D: {str(e)}")
                show_detailed_error("Erreur détaillée", e)
    
    with tabs[1]:  # Statistiques
        st.subheader("Statistiques")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Statistiques des composites")
            
            if composite_stats['count'] > 0:
                stats_df = pd.DataFrame({
                    'Paramètre': ['Nombre d\'échantillons', 'Minimum', 'Maximum', 'Moyenne', 'Médiane', 'Écart-type', 'CV'],
                    'Valeur': [
                        composite_stats['count'],
                        f"{composite_stats['min']:.3f}",
                        f"{composite_stats['max']:.3f}",
                        f"{composite_stats['mean']:.3f}",
                        f"{composite_stats['median']:.3f}",
                        f"{composite_stats['stddev']:.3f}",
                        f"{composite_stats['cv']:.3f}"
                    ]
                })
                
                st.dataframe(stats_df, hide_index=True, use_container_width=True)
                
                st.markdown("### Histogramme des composites")
                fig = plot_histogram(composite_values, f"Distribution des teneurs des composites ({col_value})", "darkblue")
                st.pyplot(fig)
            else:
                st.warning("Aucune donnée valide pour calculer les statistiques des composites.")
            
            # Statistiques de densité si disponible
            if density_method == "Variable (colonne)" and density_column:
                st.markdown("### Statistiques de densité")
                density_values = [comp.get('DENSITY') for comp in composites_data if 'DENSITY' in comp]
                if density_values:
                    density_stats = calculate_stats(density_values)
                    
                    density_stats_df = pd.DataFrame({
                        'Paramètre': ['Nombre d\'échantillons', 'Minimum', 'Maximum', 'Moyenne', 'Médiane', 'Écart-type', 'CV'],
                        'Valeur': [
                            density_stats['count'],
                            f"{density_stats['min']:.3f}",
                            f"{density_stats['max']:.3f}",
                            f"{density_stats['mean']:.3f}",
                            f"{density_stats['median']:.3f}",
                            f"{density_stats['stddev']:.3f}",
                            f"{density_stats['cv']:.3f}"
                        ]
                    })
                    
                    st.dataframe(density_stats_df, hide_index=True, use_container_width=True)
        
        with col2:
            if "estimated_blocks" in st.session_state and st.session_state.estimated_blocks:
                block_values = [block.get('value', 0) for block in st.session_state.estimated_blocks]
                block_stats = calculate_stats(block_values)
                
                st.markdown("### Statistiques du modèle de blocs")
                
                stats_df = pd.DataFrame({
                    'Paramètre': ['Nombre de blocs', 'Minimum', 'Maximum', 'Moyenne', 'Médiane', 'Écart-type', 'CV'],
                    'Valeur': [
                        block_stats['count'],
                        f"{block_stats['min']:.3f}",
                        f"{block_stats['max']:.3f}",
                        f"{block_stats['mean']:.3f}",
                        f"{block_stats['median']:.3f}",
                        f"{block_stats['stddev']:.3f}",
                        f"{block_stats['cv']:.3f}"
                    ]
                })
                
                st.dataframe(stats_df, hide_index=True, use_container_width=True)
                
                st.markdown("### Histogramme du modèle de blocs")
                fig = plot_histogram(block_values, f"Distribution des teneurs du modèle de blocs ({col_value})", "teal")
                st.pyplot(fig)
                
                # Résumé des statistiques globales
                st.markdown("### Résumé global")
                
                # Vérifier les clés nécessaires
                if (all(key in st.session_state.estimated_blocks[0] for key in ['size_x', 'size_y', 'size_z']) and 
                    block_stats['count'] > 0):
                    block_volume = st.session_state.estimated_blocks[0]['size_x'] * st.session_state.estimated_blocks[0]['size_y'] * st.session_state.estimated_blocks[0]['size_z']
                    total_volume = len(st.session_state.estimated_blocks) * block_volume
                    
                    # Calcul du tonnage avec densité variable ou constante
                    if density_method == "Variable (colonne)":
                        total_tonnage = sum(block.get('density', density_value) * block_volume for block in st.session_state.estimated_blocks)
                        avg_density = total_tonnage / total_volume if total_volume > 0 else density_value
                    else:
                        avg_density = density_value
                        total_tonnage = total_volume * avg_density
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Nombre de blocs", f"{len(st.session_state.estimated_blocks)}")
                    
                    with col2:
                        st.metric(f"Teneur moyenne {col_value}", f"{block_stats['mean']:.3f}")
                    
                    with col3:
                        st.metric("Volume total (m³)", f"{total_volume:,.0f}")
                    
                    with col4:
                        st.metric("Tonnage total (t)", f"{total_tonnage:,.0f}")
                else:
                    st.warning("Données insuffisantes pour calculer les métriques globales.")
            else:
                st.info("Veuillez d'abord créer et estimer le modèle de blocs pour afficher les statistiques.")
    
    with tabs[2]:  # Tonnage-Teneur
        st.subheader("Analyse Tonnage-Teneur")
        
        if "estimated_blocks" in st.session_state and st.session_state.estimated_blocks:
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                cutoff_method = st.selectbox(
                    "Méthode de coupure",
                    options=["above", "below", "between"],
                    format_func=lambda x: "Teneur ≥ Coupure" if x == "above" else "Teneur ≤ Coupure" if x == "below" else "Entre deux teneurs"
                )
            
            cutoff_value = None
            cutoff_min = None
            cutoff_max = None
            
            if cutoff_method == "between":
                with col2:
                    cutoff_min = st.number_input("Teneur min", min_value=0.0, value=0.5, step=0.1)
                
                with col3:
                    cutoff_max = st.number_input("Teneur max", min_value=cutoff_min, value=1.0, step=0.1)
            else:
                with col2:
                    cutoff_value = st.number_input("Teneur de coupure", min_value=0.0, value=0.5, step=0.1)
            
            with col4:
                if st.button("Calculer", type="primary"):
                    try:
                        # Détermine la méthode de densité
                        density_method_str = st.session_state.density_method if 'density_method' in st.session_state else "constant"
                        density_value_num = st.session_state.density_value if 'density_value' in st.session_state else density_value
                        
                        # Calculer les données tonnage-teneur
                        tonnage_data, plot_info = calculate_tonnage_grade(
                            st.session_state.estimated_blocks,
                            density_method_str,
                            density_value_num,
                            cutoff_method,
                            cutoff_value,
                            cutoff_min,
                            cutoff_max
                        )
                        
                        st.session_state.tonnage_data = tonnage_data
                        st.session_state.plot_info = plot_info
                    except Exception as e:
                        st.error(f"Erreur lors du calcul tonnage-teneur: {str(e)}")
                        show_detailed_error("Erreur détaillée", e)
            
            if "tonnage_data" in st.session_state:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Graphique Tonnage-Teneur
                    fig = plot_tonnage_grade(st.session_state.tonnage_data, st.session_state.plot_info)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Graphique Métal contenu
                    fig = plot_metal_content(st.session_state.tonnage_data, st.session_state.plot_info)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Tableau des résultats
                st.subheader("Résultats détaillés")
                
                # Vérifier que les données existent
                if ('plot_info' in st.session_state and 'tonnage_data' in st.session_state and
                    'cutoffs' in st.session_state.tonnage_data and 'tonnages' in st.session_state.tonnage_data and 
                    'grades' in st.session_state.tonnage_data and 'metals' in st.session_state.tonnage_data):
                    
                    if st.session_state.plot_info.get('method') == 'between':
                        # Pour la méthode between, afficher un seul résultat
                        if len(st.session_state.tonnage_data['tonnages']) > 0:
                            result_df = pd.DataFrame({
                                'Coupure': [f"{cutoff_min:.2f} - {cutoff_max:.2f}"],
                                'Tonnage (t)': [st.session_state.tonnage_data['tonnages'][0]],
                                'Teneur moyenne': [st.session_state.tonnage_data['grades'][0]],
                                'Métal contenu': [st.session_state.tonnage_data['metals'][0]]
                            })
                            st.dataframe(result_df, hide_index=True, use_container_width=True)
                        else:
                            st.warning("Aucun résultat pour cette coupure.")
                    else:
                        # Pour les méthodes above et below, afficher la courbe complète
                        result_df = pd.DataFrame({
                            'Coupure': st.session_state.tonnage_data['cutoffs'],
                            'Tonnage (t)': st.session_state.tonnage_data['tonnages'],
                            'Teneur moyenne': st.session_state.tonnage_data['grades'],
                            'Métal contenu': st.session_state.tonnage_data['metals']
                        })
                        st.dataframe(result_df, hide_index=True, use_container_width=True)
                else:
                    st.warning("Données tonnage-teneur incomplètes.")
                
                # Export des résultats
                st.subheader("Exporter les résultats")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Export Excel
                    if st.button("Exporter en Excel"):
                        try:
                            # Créer un buffer pour le fichier Excel
                            output = io.BytesIO()
                            
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                # Écrire les données tonnage-teneur
                                result_df.to_excel(writer, sheet_name='Tonnage-Teneur', index=False)
                                
                                # Ajouter une feuille pour les paramètres
                                density_info = f"Variable (colonne {density_column})" if density_method == "Variable (colonne)" else f"Constante ({density_value} t/m³)"
                                
                                param_df = pd.DataFrame({
                                    'Paramètre': [
                                        'Méthode de coupure', 
                                        'Méthode d\'estimation',
                                        'Puissance (p)',
                                        'Taille des blocs (m)',
                                        'Densité',
                                        'Date d\'exportation'
                                    ],
                                    'Valeur': [
                                        "Teneur ≥ Coupure" if cutoff_method == "above" else "Teneur ≤ Coupure" if cutoff_method == "below" else f"Entre {cutoff_min} et {cutoff_max}",
                                        'Inverse des distances',
                                        power,
                                        f"{block_size_x} × {block_size_y} × {block_size_z}",
                                        density_info,
                                        pd.Timestamp.now().strftime('%Y-%m-%d')
                                    ]
                                })
                                param_df.to_excel(writer, sheet_name='Paramètres', index=False)
                            
                            # Télécharger le fichier
                            output.seek(0)
                            st.download_button(
                                label="Télécharger Excel",
                                data=output,
                                file_name=f"{project_name.replace(' ', '_')}_tonnage_teneur_idw.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        except Exception as e:
                            st.error(f"Erreur lors de l'export Excel: {str(e)}")
                            show_detailed_error("Erreur détaillée", e)
                
                with col2:
                    # Export graphiques PNG
                    if st.button("Exporter graphiques PNG"):
                        try:
                            # Créer un buffer ZIP pour les graphiques
                            zip_buffer = io.BytesIO()
                            
                            with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
                                # Ajouter le graphique Tonnage-Teneur
                                fig = plot_tonnage_grade(st.session_state.tonnage_data, st.session_state.plot_info)
                                fig_bytes = fig.to_image(format="png", scale=2)
                                zip_file.writestr("tonnage_teneur.png", fig_bytes)
                                
                                # Ajouter le graphique Métal contenu
                                fig = plot_metal_content(st.session_state.tonnage_data, st.session_state.plot_info)
                                fig_bytes = fig.to_image(format="png", scale=2)
                                zip_file.writestr("metal_contenu.png", fig_bytes)
                            
                            # Télécharger le fichier ZIP
                            zip_buffer.seek(0)
                            st.download_button(
                                label="Télécharger graphiques PNG",
                                data=zip_buffer,
                                file_name=f"{project_name.replace(' ', '_')}_graphiques_tonnage_teneur.zip",
                                mime="application/zip"
                            )
                        except Exception as e:
                            st.error(f"Erreur lors de l'export des graphiques: {str(e)}")
                            show_detailed_error("Erreur détaillée", e)
        else:
            st.info("Veuillez d'abord créer et estimer le modèle de blocs pour effectuer l'analyse tonnage-teneur.")
            
    with tabs[3]:  # Rapport
        st.subheader("Rapport d'Estimation")
        
        if "estimated_blocks" in st.session_state and st.session_state.estimated_blocks:
            st.markdown("""
            Cette section vous permet de générer un rapport complet de l'estimation, incluant:
            - La méthodologie d'estimation
            - Les paramètres utilisés
            - Les statistiques des composites et du modèle de blocs
            - L'analyse tonnage-teneur
            - Les conclusions
            """)
            
            # Génération du rapport
            if st.button("Générer le rapport PDF", type="primary"):
                with st.spinner('Génération du rapport en cours...'):
                    try:
                        # Récupérer les paramètres stockés dans la session
                        idw_params = st.session_state.get('idw_params', {
                            'power': power,
                            'anisotropy': {'x': anisotropy_x, 'y': anisotropy_y, 'z': anisotropy_z}
                        })
                        
                        search_params = st.session_state.get('search_params', {
                            'x': search_radius_x,
                            'y': search_radius_y,
                            'z': search_radius_z,
                            'min_samples': min_samples,
                            'max_samples': max_samples
                        })
                        
                        block_sizes = st.session_state.get('block_sizes', {
                            'x': block_size_x, 
                            'y': block_size_y, 
                            'z': block_size_z
                        })
                        
                        estimated_blocks = st.session_state.estimated_blocks
                        density_method = st.session_state.get('density_method', "constant")
                        density_value = st.session_state.get('density_value', density_value)
                        density_column = st.session_state.get('density_column', density_column)
                        
                        # Récupérer les données tonnage-teneur si disponibles
                        tonnage_data = st.session_state.get('tonnage_data', None)
                        plot_info = st.session_state.get('plot_info', None)
                        
                        # Générer le rapport PDF
                        pdf_buffer = generate_estimation_report(
                            estimated_blocks,
                            composites_data,
                            idw_params,
                            search_params,
                            block_sizes,
                            tonnage_data,
                            plot_info,
                            density_method,
                            density_value,
                            density_column,
                            project_name
                        )
                        
                        if pdf_buffer:
                            st.success("Rapport généré avec succès")
                            
                            # Télécharger le PDF
                            st.download_button(
                                label="Télécharger le rapport PDF",
                                data=pdf_buffer,
                                file_name=f"{project_name.replace(' ', '_')}_rapport_estimation_IDW.pdf",
                                mime="application/pdf"
                            )
                        else:
                            st.error("Erreur lors de la génération du rapport PDF.")
                    except Exception as e:
                        st.error(f"Erreur lors de la génération du rapport PDF: {str(e)}")
                        show_detailed_error("Erreur détaillée", e)
        else:
            st.info("Veuillez d'abord créer et estimer le modèle de blocs pour générer un rapport d'estimation.")

else:
    # Affichage par défaut lorsqu'aucun fichier n'est chargé
    st.info("Veuillez charger un fichier CSV de composites dans le panneau latéral ou utiliser les données d'exemple pour commencer.")
    
    st.markdown("""
    ## Guide d'utilisation
    
    1. **Chargez un fichier CSV** contenant vos données de composites ou **utilisez les données d'exemple** de gisement de cuivre
    2. **Mappez les colonnes** pour identifier les coordonnées X, Y, Z et les valeurs de teneur
    3. **Choisissez une méthode de densité** constante ou variable (à partir d'une colonne du fichier)
    4. **Créez le modèle de blocs** en définissant la taille des blocs
    5. **Estimez par IDW** pour obtenir les teneurs des blocs
    6. **Analysez les résultats** dans les onglets Statistiques et Tonnage-Teneur
    7. **Générez un rapport PDF** complet de l'estimation
    8. **Exportez le modèle de blocs** en format CSV pour une utilisation dans d'autres logiciels
    
    ### Format du fichier CSV
    
    Le fichier CSV doit contenir au minimum les colonnes suivantes:
    - Coordonnées X, Y, Z des échantillons
    - Valeurs de teneur (par exemple, CU_PCT pour le pourcentage de cuivre)
    - Optionnellement, une colonne de densité (DENSITY)
    - Optionnellement, une colonne de domaine pour le filtrage
    
    ### Exemple de données synthétiques de cuivre
    
    Si vous activez l'option "Utiliser des données d'exemple", l'application générera un gisement de cuivre synthétique avec les caractéristiques suivantes:
    - Teneurs en cuivre entre 0.1% et 5.0%
    - Densités variables basées sur la teneur en cuivre
    - Domaines géologiques pour simuler différentes lithologies
    """)
    
    # Montrer un exemple des données synthétiques
    st.subheader("Aperçu des données d'exemple de cuivre")
    example_df = generate_synthetic_copper_data(20)
    st.dataframe(example_df.head(10))

# Footer
st.markdown("---")
st.markdown("© 2025 MineEstim - Développé par Didier Ouedraogo, P.Geo")