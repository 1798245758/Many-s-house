export default function ErrorMessage({ message, onRetry }) {
  return (
    <div className="error-message">
      <p>⚠️ {message}</p>
      {onRetry && <button className="btn btn-primary btn-sm" style={{ marginTop: 10 }} onClick={onRetry}>重试</button>}
    </div>
  )
}
