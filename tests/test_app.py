import unittest
from flask import Flask
from flask.testing import FlaskClient
from app import app, validate_file_content, load_data
import os
import tempfile
import json
from unittest.mock import patch, MagicMock

class FlaskAppTestCase(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        self.temp_dir = tempfile.TemporaryDirectory()
        app.config['UPLOAD_FOLDER'] = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_welcome_route(self):
        response = self.app.get('/welcome')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Welcome', response.data)

    def test_index_route(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Dataviz & Chat', response.data)  # Update this line to match the actual content

    def test_data_route(self):
        response = self.app.get('/data')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Data', response.data)

    def test_check_route(self):
        response = self.app.get('/check')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Check', response.data)

    def test_fix_route(self):
        response = self.app.get('/fix')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Fix', response.data)

    def test_visualisation_route(self):
        response = self.app.get('/visualisation')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Visualisation', response.data)

    def test_upload_file_no_file(self):
        response = self.app.post('/upload')
        self.assertEqual(response.status_code, 400)
        self.assertIn(b'No file part', response.data)

    def test_upload_file_no_selected_file(self):
        response = self.app.post('/upload', data={'file': (None, '')})
        self.assertEqual(response.status_code, 400)
        self.assertIn(b'No selected file', response.data)

    @patch('app.validate_file_content')
    def test_upload_file_invalid_content(self, mock_validate):
        mock_validate.return_value = False
        with open('test.txt', 'w') as f:
            f.write('Test content')
        with open('test.txt', 'rb') as f:
            response = self.app.post('/upload', data={'file': f})
        self.assertEqual(response.status_code, 400)
        self.assertIn(b'Invalid file content', response.data)
        os.remove('test.txt')

    @patch('app.validate_file_content')
    def test_upload_file_valid_content(self, mock_validate):
        mock_validate.return_value = True
        with open('test.txt', 'w') as f:
            f.write('Test content')
        with open('test.txt', 'rb') as f:
            response = self.app.post('/upload', data={'file': f})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'File uploaded successfully', response.data)
        os.remove('test.txt')

    def test_check_data_no_filename(self):
        response = self.app.post('/check_data', json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn(b'Filename is required', response.data)

    @patch('app.load_data')
    def test_check_data_load_error(self, mock_load):
        mock_load.return_value = None
        response = self.app.post('/check_data', json={'filename': 'test.csv'})
        self.assertEqual(response.status_code, 500)
        self.assertIn(b'Error loading data', response.data)

    @patch('app.load_data')
    @patch('app.analyze_data')
    def test_check_data_success(self, mock_analyze, mock_load):
        mock_load.return_value = MagicMock()
        mock_analyze.return_value = {'result': 'test'}
        response = self.app.post('/check_data', json={'filename': 'test.csv'})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Data quality check completed', response.data)

    def test_apply_fixes_no_filename(self):
        response = self.app.post('/apply_fixes', json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn(b'Filename is required', response.data)

    @patch('app.load_data')
    def test_apply_fixes_load_error(self, mock_load):
        mock_load.return_value = None
        response = self.app.post('/apply_fixes', json={'filename': 'test.csv'})
        self.assertEqual(response.status_code, 500)
        self.assertIn(b'Error loading data', response.data)

    @patch('app.load_data')
    @patch('app.apply_fixes_to_data')
    def test_apply_fixes_success(self, mock_fixes, mock_load):
        mock_load.return_value = MagicMock()
        mock_fixes.return_value = (MagicMock(), 'Fixes applied')
        response = self.app.post('/apply_fixes', json={'filename': 'test.csv'})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Fixes applied', response.data)

    def test_generate_plots_no_filename(self):
        response = self.app.post('/generate_plots', json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn(b'Filename and API key are required', response.data)

    @patch('app.load_data')
    def test_generate_plots_load_error(self, mock_load):
        mock_load.return_value = None
        response = self.app.post('/generate_plots', json={'filename': 'test.csv', 'selectedModel': 'gemini', 'apiKey': 'test'})
        self.assertEqual(response.status_code, 500)
        self.assertIn(b'Error loading data', response.data)

    @patch('app.load_data')
    @patch('app.get_plot_suggestion_from_gemini')
    def test_generate_plots_success(self, mock_suggestion, mock_load):
        mock_load.return_value = MagicMock()
        mock_suggestion.return_value = {'suggestion': 'test'}
        response = self.app.post('/generate_plots', json={'filename': 'test.csv', 'selectedModel': 'gemini', 'apiKey': 'test'})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Plots generated', response.data)

    def test_get_interpretation_missing_data(self):
        response = self.app.post('/get_interpretation', json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn(b'Missing data', response.data)

    @patch('app.load_data')
    def test_get_interpretation_load_error(self, mock_load):
        mock_load.return_value = None
        response = self.app.post('/get_interpretation', json={'filename': 'test.csv', 'selectedModel': 'gemini', 'apiKey': 'test', 'suggestion': 'test'})
        self.assertEqual(response.status_code, 500)
        self.assertIn(b'Error loading data', response.data)

    @patch('app.load_data')
    @patch('app.generate_graph_interpretation_gemini')
    def test_get_interpretation_success(self, mock_interpretation, mock_load):
        mock_load.return_value = MagicMock()
        mock_interpretation.return_value = 'Interpretation'
        response = self.app.post('/get_interpretation', json={'filename': 'test.csv', 'selectedModel': 'gemini', 'apiKey': 'test', 'suggestion': 'test'})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Interpretation', response.data)

    def test_graph_chat_missing_data(self):
        response = self.app.post('/graph_chat', json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn(b'Missing data', response.data)

    @patch('app.load_data')
    def test_graph_chat_load_error(self, mock_load):
        mock_load.return_value = None
        response = self.app.post('/graph_chat', json={'filename': 'test.csv', 'selectedModel': 'gemini', 'apiKey': 'test', 'message': 'test', 'image': 'test'})
        self.assertEqual(response.status_code, 500)
        self.assertIn(b'Error loading data', response.data)

    @patch('app.load_data')
    @patch('app.handle_graph_communication_gemini')
    def test_graph_chat_success(self, mock_communication, mock_load):
        mock_load.return_value = MagicMock()
        mock_communication.return_value = 'Response'
        response = self.app.post('/graph_chat', json={'filename': 'test.csv', 'selectedModel': 'gemini', 'apiKey': 'test', 'message': 'test', 'image': 'test'})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Response', response.data)

    def test_uploaded_file(self):
        with open(os.path.join(self.temp_dir.name, 'test.txt'), 'w') as f:
            f.write('Test content')
        response = self.app.get('/uploads/test.txt')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Test content', response.data)

if __name__ == '__main__':
    unittest.main()
