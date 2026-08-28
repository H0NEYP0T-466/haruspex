/**
 * HARUSPEX API Service Layer
 * Communicates with the FastAPI backend at /api
 */

const API_BASE = 'http://localhost:8014/api';

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
    attribution_method?: string;
    output_space?: string;
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

export interface ModelGroupInfo {
  models: string[];
  default: string | null;
}

export interface ModelInfo {
  groups: Record<string, ModelGroupInfo>;
  active_group: string | null;
  saved: { name: string; path: string; format: string }[];
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
  model_groups: Record<string, string[]>;
  active_group: string | null;
  version: string;
}

export interface MultiModelPredictionResponse {
  sample_id: string;
  models_tested: number;
  summary: {
    consensus: 'malware' | 'benign' | 'tie';
    malware_votes: number;
    benign_votes: number;
  };
  results: {
    model_group: string;
    model_name: string;
    prediction: string;
    confidence: number;
    malware_probability: number;
    in_distribution?: boolean;
  }[];
  forensic_summary: Record<string, unknown>;
  ioc_count: number;
  true_label?: string;
  dataset_group?: string;
  notice?: string;
}

export interface UploadedForensicsResponse {
  sample_id: string;
  ml_verdict_available: boolean;
  ml_verdict: {
    available: boolean;
    consensus?: string;
    malware_votes?: number;
    benign_votes?: number;
    per_model?: {
      model_group: string;
      model_name: string;
      prediction: string;
      confidence: number;
    }[];
    error?: string;
  } | null;
  shap_features: ShapFeature[];
  lime_features: LimeFeature[];
  feature_vectorizer: string;
  vector_schema: string;
  forensic_summary: Record<string, unknown>;
  ioc_count: number;
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

  async predictFromVector(
    features: number[],
    modelName?: string,
    modelGroup?: string,
  ): Promise<PredictionReport> {
    return this.request<PredictionReport>('/predict/vector', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        features,
        model_name: modelName,
        model_group: modelGroup,
      }),
    });
  }

  async predictFromFile(
    file: File,
    modelName?: string,
    modelGroup?: string,
  ): Promise<PredictionReport> {
    const formData = new FormData();
    formData.append('file', file);
    if (modelName) {
      formData.append('model_name', modelName);
    }
    if (modelGroup) {
      formData.append('model_group', modelGroup);
    }
    return this.request<PredictionReport>('/predict', {
      method: 'POST',
      body: formData,
    });
  }

  async predictAllFromFile(file: File): Promise<MultiModelPredictionResponse> {
    const formData = new FormData();
    formData.append('file', file);
    return this.request<MultiModelPredictionResponse>('/predict/all', {
      method: 'POST',
      body: formData,
    });
  }

  async predictAllFromVector(features: number[]): Promise<MultiModelPredictionResponse> {
    return this.request<MultiModelPredictionResponse>('/predict/all', {
      method: 'POST',
      body: new URLSearchParams({
        features_json: JSON.stringify(features),
      }),
    });
  }

  async analyzeUploadedFileForensics(file: File): Promise<UploadedForensicsResponse> {
    const formData = new FormData();
    formData.append('file', file);
    return this.request<UploadedForensicsResponse>('/predict/upload-forensics', {
      method: 'POST', body: formData,
    });
  }

  async explainRandomAll(modelGroup?: string): Promise<MultiModelPredictionResponse & {
    sample_index: number; dataset_group: string; true_label: string; notice: string;
  }> {
    return this.request('/explain/random/all', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_group: modelGroup }),
    });
  }

  async explainSample(
    sampleIndex: number,
    modelName?: string,
    modelGroup?: string,
    numFeatures = 20,
  ): Promise<PredictionReport> {
    return this.request<PredictionReport>('/explain', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sample_index: sampleIndex,
        model_name: modelName,
        model_group: modelGroup,
        num_features: numFeatures,
      }),
    });
  }

  async explainRandom(
    modelName?: string,
    modelGroup?: string,
    labelFilter?: string,
    numFeatures = 20,
  ): Promise<PredictionReport> {
    return this.request<PredictionReport>('/explain/random', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model_name: modelName,
        model_group: modelGroup,
        label_filter: labelFilter,
        num_features: numFeatures,
      }),
    });
  }

  async getDatasetInfo(dataset?: string): Promise<DatasetInfo> {
    const params = dataset ? `?dataset=${dataset}` : '';
    return this.request<DatasetInfo>(`/dataset/info${params}`);
  }

  async listSamples(
    filters?: { family?: string; category?: string; label?: string },
    offset = 0,
    limit = 50,
    dataset?: string,
  ): Promise<SampleListResponse> {
    const params = new URLSearchParams();
    params.set('offset', String(offset));
    params.set('limit', String(limit));
    if (filters?.family) params.set('family', filters.family);
    if (filters?.category) params.set('category', filters.category);
    if (filters?.label) params.set('label', filters.label);
    if (dataset) params.set('dataset', dataset);
    return this.request<SampleListResponse>(`/dataset/samples?${params}`);
  }
}

export const api = new HaruspexAPI();
export default api;
