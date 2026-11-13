const SIZE_ERROR = 'This upload was rejected as too large. Choose an image up to 8 MB.';

export async function readInferenceResponse(response: Response): Promise<any> {
  let result: any;
  try {
    result = JSON.parse(await response.text());
  } catch {
    if (response.status === 413) throw new Error(SIZE_ERROR);
    if (response.status === 429) throw new Error('The model service is busy. Please try again shortly.');
    throw new Error('The model service returned an unexpected response. Please try again.');
  }
  if (!response.ok) {
    const explanation = typeof result?.error === 'string' ? result.error
      : typeof result?.detail === 'string' ? result.detail : null;
    throw new Error(explanation ?? (response.status === 413 ? SIZE_ERROR
      : 'The model service could not process this image. Please try again.'));
  }
  if (!result?.geometry || !result?.methods || !Number.isFinite(result.seconds) ||
      !['ggf', 'slics', 'supervised_vanilla', 'vanilla'].every(method =>
        result.methods[method]?.norms?.length === 75 &&
        result.methods[method]?.positions?.length === 75 &&
        result.methods[method]?.scores?.length === 75)) {
    throw new Error('The model service returned an incomplete result. Please try again.');
  }
  return result;
}
