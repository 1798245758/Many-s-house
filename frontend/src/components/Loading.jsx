export default function Loading({ text = '加载中' }) {
  return (
    <div className="loading-wrap">
      <div className="spinner" />
      <div className="loading-dots muted">{text}<span>.</span><span>.</span><span>.</span></div>
    </div>
  )
}
