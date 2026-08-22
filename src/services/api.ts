/**
 * HARUSPEX API Service Layer
 * Communicates with the FastAPI backend at /api
 */

const API_BASE = '/api';

export interface PredictionReport {
  sample: {
    id: string;
    sha256: string;
    family: string;
    category: string;
  };
  prediction: {
    verdict: string;
    confidence: number;
    risk_score: number;
    model: string;
  };
  evidence_summary: {
    total: number;
    correlated: number;
    ai_only: number;
    forensic_only: number;
  };
  evidence_items: EvidenceItem[];
  xai: {
    top_shap_features: ShapFeature[];
    lime_features: LimeFeature[];
  };
  forensics: Record<string, unknown>;
  ioc_count: number;
  metadata: {
    timestamp: string;
    haruspex_version: string;
  };
  true_label?: string;
  sample_index?: number;
}

export interface EvidenceItem {
  category: string;
  indicator: string;
  severity: number;
  source: string;
  confidence: number;
  feature_name?: string;
  shap_value?: number;
  matched_forensic?: string;
}

export interface ShapFeature {
  index: number;
  feature_name: string;
  shap_value: number;
  feature_value: number;
}

export interface LimeFeature {
  feature_description: string;
  feature_name: string;
  weight: number;
}

export interface ModelInfo {
  loaded: string[];
  saved: { name: string; path: string; format: string }[];
  default: string | null;
}

export interface DatasetInfo {
  total_samples: number;
  n_features: number;
  n_benign: number;
  n_malware: number;
  n_families: number;
  families: string[];
  top_families: { name: string; count: number }[];
  categories: Record<string, number>;
}

export interface SampleListResponse {
  total: number;
  offset: number;
  limit: number;
  samples: DatasetSample[];
}

export interface DatasetSample {
  index: number;
  sha256: string;
  family: string;
  category: string;
  label: string;
  timestamp: string;
}

export interface HealthResponse {
  status: string;
  models_loaded: string[];
  version: string;
}

class HaruspexAPI {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(path: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      ...options,
      headers: {
        ...options?.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }

    return response.json();
  }

  async health(): Promise<HealthResponse> {
    return this.request<HealthResponse>('/health');
  }

  async getModels(): Promise<ModelInfo> {
    return this.request<ModelInfo>('/models');
  }

  async predictFromVector(features: number[], modelName?: string): Promise<PredictionReport> {
    return this.request<PredictionReport>('/predict/vector', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features, model_name: modelName }),
    });
  }

  async predictFromFile(file: File, modelName?: string): Promise<PredictionReport> {
    const formData = new FormData();
    formData.append('file', file);
    if (modelName) {
      formData.append('model_name', modelName);
    }
    return this.request<PredictionReport>('/predict', {
      method: 'POST',
      body: formData,
    });
  }

  async explainSample(sampleIndex: number, modelName?: string, numFeatures = 20): Promise<PredictionReport> {
    return this.request<PredictionReport>('/explain', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sample_index: sampleIndex,
        model_name: modelName,
        num_features: numFeatures,
      }),
    });
  }

  async getDatasetInfo(): Promise<DatasetInfo> {
    return this.request<DatasetInfo>('/dataset/info');
  }

  async listSamples(
    filters?: { family?: string; category?: string; label?: string },
    offset = 0,
    limit = 50,
  ): Promise<SampleListResponse> {
    const params = new URLSearchParams();
    params.set('offset', String(offset));
    params.set('limit', String(limit));
    if (filters?.family) params.set('family', filters.family);
    if (filters?.category) params.set('category', filters.category);
    if (filters?.label) params.set('label', filters.label);
    return this.request<SampleListResponse>(`/dataset/samples?${params}`);
  }
}

export const api = new HaruspexAPI();
export default api;
